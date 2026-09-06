// Tile filter and late materialization kernels for the operator at a time path. The fused JIT
// path generates the same shape per pipeline; these are the always available fallbacks.
#include <cstdint>
#include <cuda_runtime.h>

namespace aster::cuda {

__device__ __forceinline__ bool BitGet(const uint8_t* bits, uint32_t i) { return bits ? ((bits[i >> 3] >> (i & 7)) & 1) : true; }

// Compacts surviving row indices with a block wide ballot + prefix, one atomic per warp.
template <typename Pred>
__device__ void CompactRows(uint32_t num_rows, Pred pred, uint32_t* out_rows, uint32_t* out_count) {
  for (uint32_t base = blockIdx.x * blockDim.x; base < num_rows; base += gridDim.x * blockDim.x) {
    uint32_t row = base + threadIdx.x;
    bool keep = row < num_rows && pred(row);
    unsigned mask = __ballot_sync(0xffffffffu, keep);
    unsigned lane = threadIdx.x & 31;
    unsigned prefix = __popc(mask & ((1u << lane) - 1));
    unsigned total = __popc(mask);
    uint32_t warp_base = 0;
    if (lane == 0 && total) warp_base = atomicAdd(out_count, total);
    warp_base = __shfl_sync(0xffffffffu, warp_base, 0);
    if (keep) out_rows[warp_base + prefix] = row;
  }
}

template <typename T>
struct RangePred {
  const T* values; const uint8_t* validity; T lo, hi; bool lo_inclusive, hi_inclusive;
  __device__ bool operator()(uint32_t r) const {
    if (!BitGet(validity, r)) return false;
    T v = values[r];
    bool a = lo_inclusive ? v >= lo : v > lo;
    bool b = hi_inclusive ? v <= hi : v < hi;
    return a && b;
  }
};

struct CodeEqPred {
  const int32_t* codes; const uint8_t* validity; int32_t code; bool negate;
  __device__ bool operator()(uint32_t r) const { return BitGet(validity, r) && ((codes[r] == code) != negate); }
};

struct PackedLtPred {
  const uint8_t* packed; const uint8_t* validity; uint32_t bit_width; uint64_t limit;
  __device__ bool operator()(uint32_t r) const {
    if (!BitGet(validity, r)) return false;
    uint64_t bit = uint64_t(r) * bit_width, byte = bit >> 3; uint32_t shift = bit & 7;
    uint64_t word = 0;
#pragma unroll
    for (int k = 0; k < 9; ++k) word |= uint64_t(packed[byte + k]) << (8 * k);
    uint64_t v = (word >> shift) & (bit_width == 64 ? ~0ull : ((1ull << bit_width) - 1));
    return v < limit;
  }
};

template <typename T>
__global__ void RangeFilterKernel(RangePred<T> p, uint32_t n, uint32_t* rows, uint32_t* count) { CompactRows(n, p, rows, count); }
__global__ void CodeEqFilterKernel(CodeEqPred p, uint32_t n, uint32_t* rows, uint32_t* count) { CompactRows(n, p, rows, count); }
__global__ void PackedLtFilterKernel(PackedLtPred p, uint32_t n, uint32_t* rows, uint32_t* count) { CompactRows(n, p, rows, count); }

// Intersects a selection with a second predicate: reads only selected rows (late materialization).
__global__ void RefineSelectionKernel(const uint32_t* in_rows, uint32_t in_count, const uint8_t* mask, uint32_t* out_rows, uint32_t* out_count) {
  CompactRows(in_count, [&](uint32_t i) { return mask[in_rows[i]] != 0; }, out_rows, out_count);
  // out_rows now holds indices into in_rows; a second pass gathers the row ids.
}

template <typename T>
__global__ void GatherKernel(const T* src, const uint32_t* rows, uint32_t n, T* dst) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[rows[i]];
}

__global__ void GatherStringOffsetsKernel(const int32_t* offsets, const uint32_t* rows, uint32_t n, int32_t* lengths) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) lengths[i] = offsets[rows[i] + 1] - offsets[rows[i]];
}

__global__ void GatherStringBytesKernel(const int32_t* offsets, const char* bytes, const uint32_t* rows, const int32_t* out_offsets, uint32_t n, char* out) {
  for (uint32_t i = blockIdx.x; i < n; i += gridDim.x) {
    int32_t s = offsets[rows[i]], len = offsets[rows[i] + 1] - s;
    for (int32_t k = threadIdx.x; k < len; k += blockDim.x) out[out_offsets[i] + k] = bytes[s + k];
  }
}

extern "C" {
cudaError_t aster_filter_range_i64(const int64_t* v, const uint8_t* valid, int64_t lo, int64_t hi, bool li, bool hi_inc, uint32_t n, uint32_t* rows, uint32_t* count, cudaStream_t s) {
  RangeFilterKernel<int64_t><<<(n + 255) / 256, 256, 0, s>>>(RangePred<int64_t>{v, valid, lo, hi, li, hi_inc}, n, rows, count);
  return cudaGetLastError();
}
cudaError_t aster_filter_range_f64(const double* v, const uint8_t* valid, double lo, double hi, bool li, bool hi_inc, uint32_t n, uint32_t* rows, uint32_t* count, cudaStream_t s) {
  RangeFilterKernel<double><<<(n + 255) / 256, 256, 0, s>>>(RangePred<double>{v, valid, lo, hi, li, hi_inc}, n, rows, count);
  return cudaGetLastError();
}
cudaError_t aster_filter_code_eq(const int32_t* codes, const uint8_t* valid, int32_t code, bool negate, uint32_t n, uint32_t* rows, uint32_t* count, cudaStream_t s) {
  CodeEqFilterKernel<<<(n + 255) / 256, 256, 0, s>>>(CodeEqPred{codes, valid, code, negate}, n, rows, count);
  return cudaGetLastError();
}
cudaError_t aster_filter_packed_lt(const uint8_t* packed, const uint8_t* valid, uint32_t bw, uint64_t limit, uint32_t n, uint32_t* rows, uint32_t* count, cudaStream_t s) {
  PackedLtFilterKernel<<<(n + 255) / 256, 256, 0, s>>>(PackedLtPred{packed, valid, bw, limit}, n, rows, count);
  return cudaGetLastError();
}
cudaError_t aster_gather_i64(const int64_t* src, const uint32_t* rows, uint32_t n, int64_t* dst, cudaStream_t s) {
  GatherKernel<int64_t><<<(n + 255) / 256, 256, 0, s>>>(src, rows, n, dst);
  return cudaGetLastError();
}
cudaError_t aster_gather_f64(const double* src, const uint32_t* rows, uint32_t n, double* dst, cudaStream_t s) {
  GatherKernel<double><<<(n + 255) / 256, 256, 0, s>>>(src, rows, n, dst);
  return cudaGetLastError();
}
cudaError_t aster_gather_string_lengths(const int32_t* offsets, const uint32_t* rows, uint32_t n, int32_t* lengths, cudaStream_t s) {
  GatherStringOffsetsKernel<<<(n + 255) / 256, 256, 0, s>>>(offsets, rows, n, lengths);
  return cudaGetLastError();
}
cudaError_t aster_gather_string_bytes(const int32_t* offsets, const char* bytes, const uint32_t* rows, const int32_t* out_offsets, uint32_t n, char* out, cudaStream_t s) {
  GatherStringBytesKernel<<<n ? n : 1, 128, 0, s>>>(offsets, bytes, rows, out_offsets, n, out);
  return cudaGetLastError();
}
}

}  // namespace aster::cuda
