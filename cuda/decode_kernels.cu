// Hand written decode kernels for the operator at a time path. One launch decodes one column chunk
// straight from its encoded page; the JIT fused path inlines the same logic per tile.
#include <cstdint>
#include <cuda_runtime.h>

namespace aster::cuda {

struct ChunkHeader {
  uint16_t magic; uint8_t encoding, inner, type_id, codec; uint16_t reserved;
  uint32_t type_width, num_rows, null_count, payload_bytes, raw_bytes;
};

__device__ __forceinline__ uint64_t Unpack(const uint8_t* p, uint32_t idx, uint32_t bw) {
  if (bw == 0) return 0;
  uint64_t bit = uint64_t(idx) * bw;
  uint64_t byte = bit >> 3;
  uint32_t shift = bit & 7;
  uint64_t word = 0;
#pragma unroll
  for (int k = 0; k < 9; ++k) word |= uint64_t(p[byte + k]) << (8 * k);
  uint64_t mask = bw == 64 ? ~0ull : ((1ull << bw) - 1);
  return (word >> shift) & mask;
}

// Verifies CRC32C on device while data is read, so GDS pages never need a host pass.
__device__ uint32_t Crc32cBlock(const uint8_t* data, size_t len) {
  uint32_t crc = ~0u;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
  }
  return ~crc;
}

__global__ void VerifyChecksumKernel(const uint8_t* page, uint32_t bytes, uint32_t expected, int* ok) {
  // Chunked parallel CRC needs combine tables; a single block verifies sequentially per 4 KB slice
  // and reduces equality of the slice CRC chain on the host side. Simplified: block 0 does it all.
  if (blockIdx.x == 0 && threadIdx.x == 0) *ok = Crc32cBlock(page, bytes) == expected;
}

template <typename T>
__global__ void DecodeForKernel(const uint8_t* page, T* out) {
  const ChunkHeader* h = reinterpret_cast<const ChunkHeader*>(page);
  const uint8_t* p = page + sizeof(ChunkHeader);
  int64_t ref = *reinterpret_cast<const int64_t*>(p);
  uint8_t bw = p[8];
  p += 16;
  if (h->null_count) p += (h->num_rows + 7) / 8;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h->num_rows; i += gridDim.x * blockDim.x)
    out[i] = static_cast<T>(ref + static_cast<int64_t>(Unpack(p, i, bw)));
}

__global__ void DecodeDictionaryCodesKernel(const uint8_t* page, int32_t* codes) {
  const ChunkHeader* h = reinterpret_cast<const ChunkHeader*>(page);
  const uint8_t* p = page + sizeof(ChunkHeader);
  uint32_t count = *reinterpret_cast<const uint32_t*>(p);
  uint8_t cw = p[4];
  const int32_t* off = reinterpret_cast<const int32_t*>(p + 8);
  p += 8 + (count + 1) * 4 + off[count];
  if (h->null_count) p += (h->num_rows + 7) / 8;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h->num_rows; i += gridDim.x * blockDim.x)
    codes[i] = cw == 1 ? p[i] : cw == 2 ? reinterpret_cast<const uint16_t*>(p)[i] : reinterpret_cast<const int32_t*>(p)[i];
}

// Run length: expands runs with a prefix sum over run lengths (one block per 1024 runs).
__global__ void RunOffsetsKernel(const uint32_t* lengths, uint32_t num_runs, uint32_t* offsets) {
  __shared__ uint32_t s[1024];
  uint32_t t = threadIdx.x;
  uint32_t base = blockIdx.x * 1024;
  s[t] = base + t < num_runs ? lengths[base + t] : 0;
  __syncthreads();
  for (uint32_t d = 1; d < 1024; d <<= 1) {
    uint32_t v = t >= d ? s[t - d] : 0;
    __syncthreads();
    s[t] += v;
    __syncthreads();
  }
  if (base + t < num_runs) offsets[base + t] = s[t];
}

template <typename T>
__global__ void ExpandRunsKernel(const uint8_t* run_values, uint8_t width, const uint32_t* run_ends, uint32_t num_runs, uint32_t num_rows, T* out) {
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += gridDim.x * blockDim.x) {
    uint32_t lo = 0, hi = num_runs;
    while (lo + 1 < hi) { uint32_t mid = (lo + hi) / 2; if (run_ends[mid - 1] <= row) lo = mid; else hi = mid; }
    int64_t v = 0;
    switch (width) {
      case 1: v = reinterpret_cast<const int8_t*>(run_values)[lo]; break;
      case 2: v = reinterpret_cast<const int16_t*>(run_values)[lo]; break;
      case 4: v = reinterpret_cast<const int32_t*>(run_values)[lo]; break;
      default: v = reinterpret_cast<const int64_t*>(run_values)[lo]; break;
    }
    out[row] = static_cast<T>(v);
  }
}

template <typename T>
__global__ void DecodeDeltaKernel(const uint8_t* page, T* out) {
  // Serial prefix within a block, block carries combined by a second pass on the host stream.
  const ChunkHeader* h = reinterpret_cast<const ChunkHeader*>(page);
  const uint8_t* p = page + sizeof(ChunkHeader);
  int64_t first = *reinterpret_cast<const int64_t*>(p);
  int64_t min_d = *reinterpret_cast<const int64_t*>(p + 8);
  uint8_t bw = p[16];
  p += 24;
  if (h->null_count) p += (h->num_rows + 7) / 8;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t cur = first;
    if (h->num_rows) out[0] = static_cast<T>(cur);
    for (uint32_t i = 1; i < h->num_rows; ++i) { cur += min_d + static_cast<int64_t>(Unpack(p, i - 1, bw)); out[i] = static_cast<T>(cur); }
  }
}

extern "C" {
cudaError_t aster_decode_for_i64(const uint8_t* page, int64_t* out, uint32_t rows, cudaStream_t s) {
  DecodeForKernel<int64_t><<<(rows + 255) / 256, 256, 0, s>>>(page, out);
  return cudaGetLastError();
}
cudaError_t aster_decode_for_i32(const uint8_t* page, int32_t* out, uint32_t rows, cudaStream_t s) {
  DecodeForKernel<int32_t><<<(rows + 255) / 256, 256, 0, s>>>(page, out);
  return cudaGetLastError();
}
cudaError_t aster_decode_dictionary_codes(const uint8_t* page, int32_t* codes, uint32_t rows, cudaStream_t s) {
  DecodeDictionaryCodesKernel<<<(rows + 255) / 256, 256, 0, s>>>(page, codes);
  return cudaGetLastError();
}
cudaError_t aster_expand_runs_i64(const uint8_t* values, uint8_t width, const uint32_t* run_ends, uint32_t runs, uint32_t rows, int64_t* out, cudaStream_t s) {
  ExpandRunsKernel<int64_t><<<(rows + 255) / 256, 256, 0, s>>>(values, width, run_ends, runs, rows, out);
  return cudaGetLastError();
}
cudaError_t aster_run_offsets(const uint32_t* lengths, uint32_t runs, uint32_t* offsets, cudaStream_t s) {
  RunOffsetsKernel<<<(runs + 1023) / 1024, 1024, 0, s>>>(lengths, runs, offsets);
  return cudaGetLastError();
}
cudaError_t aster_decode_delta_i64(const uint8_t* page, int64_t* out, cudaStream_t s) {
  DecodeDeltaKernel<int64_t><<<1, 1, 0, s>>>(page, out);
  return cudaGetLastError();
}
cudaError_t aster_verify_checksum(const uint8_t* page, uint32_t bytes, uint32_t expected, int* ok, cudaStream_t s) {
  VerifyChecksumKernel<<<1, 1, 0, s>>>(page, bytes, expected, ok);
  return cudaGetLastError();
}
}

}  // namespace aster::cuda
