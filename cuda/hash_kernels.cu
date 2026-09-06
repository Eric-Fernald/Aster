// Hash join build/probe and partial aggregate kernels. Open addressing, linear probing, keys stored
// as (hash | 1) so zero marks empty. Shared dictionary joins hash the int32 code, never the string.
#include <cstdint>
#include <cuda_runtime.h>

namespace aster::cuda {

__device__ __forceinline__ uint64_t Mix(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
  return x;
}

__global__ void HashKeysI64Kernel(const int64_t* keys, const uint8_t* validity, uint32_t n, uint64_t* out, bool combine) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    bool valid = validity ? ((validity[i >> 3] >> (i & 7)) & 1) : true;
    uint64_t h = valid ? Mix(static_cast<uint64_t>(keys[i])) : 0xdeadbeefull;
    out[i] = combine ? (out[i] ^ (h + 0x9e3779b97f4a7c15ull + (out[i] << 6) + (out[i] >> 2))) : h;
  }
}

__global__ void HashKeysI32Kernel(const int32_t* keys, const uint8_t* validity, uint32_t n, uint64_t* out, bool combine) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    bool valid = validity ? ((validity[i >> 3] >> (i & 7)) & 1) : true;
    uint64_t h = valid ? Mix(static_cast<uint64_t>(static_cast<int64_t>(keys[i]))) : 0xdeadbeefull;
    out[i] = combine ? (out[i] ^ (h + 0x9e3779b97f4a7c15ull + (out[i] << 6) + (out[i] >> 2))) : h;
  }
}

// Build: insert (hash, row) pairs. Duplicates chain through `next` so multi match joins work.
__global__ void BuildTableKernel(const uint64_t* hashes, uint32_t n, uint64_t* slots, int32_t* slot_rows, int32_t* next, uint32_t mask) {
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    uint64_t key = hashes[r] | 1ull;
    uint32_t s = static_cast<uint32_t>(key & mask);
    for (;;) {
      unsigned long long prev = atomicCAS(reinterpret_cast<unsigned long long*>(&slots[s]), 0ull, key);
      if (prev == 0ull || prev == key) {
        int32_t old = atomicExch(&slot_rows[s], static_cast<int32_t>(r));
        next[r] = old;
        break;
      }
      s = (s + 1) & mask;
    }
  }
}

// Probe: count matches per probe row (pass 1) then emit pairs (pass 2) after an exclusive scan.
__global__ void ProbeCountKernel(const uint64_t* probe_hashes, uint32_t n, const uint64_t* slots, const int32_t* slot_rows, const int32_t* next, uint32_t mask, uint32_t* counts) {
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    uint64_t key = probe_hashes[r] | 1ull;
    uint32_t s = static_cast<uint32_t>(key & mask);
    uint32_t c = 0;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
      uint64_t k = slots[s];
      if (k == 0ull) break;
      if (k == key) { for (int32_t b = slot_rows[s]; b >= 0; b = next[b]) ++c; break; }
      s = (s + 1) & mask;
    }
    counts[r] = c;
  }
}

__global__ void ProbeEmitKernel(const uint64_t* probe_hashes, uint32_t n, const uint64_t* slots, const int32_t* slot_rows, const int32_t* next, uint32_t mask,
                                const uint32_t* offsets, uint32_t* out_probe, uint32_t* out_build) {
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    uint64_t key = probe_hashes[r] | 1ull;
    uint32_t s = static_cast<uint32_t>(key & mask);
    uint32_t pos = offsets[r];
    for (uint32_t probes = 0; probes <= mask; ++probes) {
      uint64_t k = slots[s];
      if (k == 0ull) break;
      if (k == key) { for (int32_t b = slot_rows[s]; b >= 0; b = next[b]) { out_probe[pos] = r; out_build[pos] = static_cast<uint32_t>(b); ++pos; } break; }
      s = (s + 1) & mask;
    }
  }
}

// Semi/anti: one flag per probe row.
__global__ void ProbeExistsKernel(const uint64_t* probe_hashes, uint32_t n, const uint64_t* slots, uint32_t mask, uint8_t* exists) {
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    uint64_t key = probe_hashes[r] | 1ull;
    uint32_t s = static_cast<uint32_t>(key & mask);
    uint8_t found = 0;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
      uint64_t k = slots[s];
      if (k == 0ull) break;
      if (k == key) { found = 1; break; }
      s = (s + 1) & mask;
    }
    exists[r] = found;
  }
}

// Partial aggregate: sum/count/min/max per group hash with atomics; finalize merges on host or with a second launch.
__global__ void PartialAggregateKernel(const uint64_t* group_hashes, const double* values, const uint32_t* run_lengths, uint32_t n,
                                       uint64_t* keys, double* sums, uint32_t* counts, double* mins, double* maxs, uint32_t slots_mask) {
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    uint64_t key = group_hashes[r] | 1ull;
    uint32_t s = static_cast<uint32_t>(key & slots_mask);
    for (;;) {
      unsigned long long prev = atomicCAS(reinterpret_cast<unsigned long long*>(&keys[s]), 0ull, key);
      if (prev == 0ull || prev == key) break;
      s = (s + 1) & slots_mask;
    }
    uint32_t w = run_lengths ? run_lengths[r] : 1;  // run level aggregation multiplies by the run length
    double v = values ? values[r] : 1.0;
    atomicAdd(&sums[s], v * w);
    atomicAdd(&counts[s], w);
    if (mins) { unsigned long long* a = reinterpret_cast<unsigned long long*>(&mins[s]); unsigned long long o = *a, x; do { x = o; if (__longlong_as_double(x) <= v) break; o = atomicCAS(a, x, __double_as_longlong(v)); } while (o != x); }
    if (maxs) { unsigned long long* a = reinterpret_cast<unsigned long long*>(&maxs[s]); unsigned long long o = *a, x; do { x = o; if (__longlong_as_double(x) >= v) break; o = atomicCAS(a, x, __double_as_longlong(v)); } while (o != x); }
  }
}

// Radix partition assignment for spilling builds and multi GPU shuffles.
__global__ void RadixPartitionKernel(const uint64_t* hashes, uint32_t n, uint32_t bits, uint32_t* partition) {
  uint32_t mask = (1u << bits) - 1;
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) partition[r] = static_cast<uint32_t>((hashes[r] >> 40) & mask);
}

extern "C" {
cudaError_t aster_hash_keys_i64(const int64_t* keys, const uint8_t* valid, uint32_t n, uint64_t* out, bool combine, cudaStream_t s) {
  HashKeysI64Kernel<<<(n + 255) / 256, 256, 0, s>>>(keys, valid, n, out, combine);
  return cudaGetLastError();
}
cudaError_t aster_hash_keys_i32(const int32_t* keys, const uint8_t* valid, uint32_t n, uint64_t* out, bool combine, cudaStream_t s) {
  HashKeysI32Kernel<<<(n + 255) / 256, 256, 0, s>>>(keys, valid, n, out, combine);
  return cudaGetLastError();
}
cudaError_t aster_build_table(const uint64_t* hashes, uint32_t n, uint64_t* slots, int32_t* slot_rows, int32_t* next, uint32_t mask, cudaStream_t s) {
  BuildTableKernel<<<(n + 255) / 256, 256, 0, s>>>(hashes, n, slots, slot_rows, next, mask);
  return cudaGetLastError();
}
cudaError_t aster_probe_count(const uint64_t* ph, uint32_t n, const uint64_t* slots, const int32_t* slot_rows, const int32_t* next, uint32_t mask, uint32_t* counts, cudaStream_t s) {
  ProbeCountKernel<<<(n + 255) / 256, 256, 0, s>>>(ph, n, slots, slot_rows, next, mask, counts);
  return cudaGetLastError();
}
cudaError_t aster_probe_emit(const uint64_t* ph, uint32_t n, const uint64_t* slots, const int32_t* slot_rows, const int32_t* next, uint32_t mask, const uint32_t* offsets, uint32_t* op, uint32_t* ob, cudaStream_t s) {
  ProbeEmitKernel<<<(n + 255) / 256, 256, 0, s>>>(ph, n, slots, slot_rows, next, mask, offsets, op, ob);
  return cudaGetLastError();
}
cudaError_t aster_probe_exists(const uint64_t* ph, uint32_t n, const uint64_t* slots, uint32_t mask, uint8_t* exists, cudaStream_t s) {
  ProbeExistsKernel<<<(n + 255) / 256, 256, 0, s>>>(ph, n, slots, mask, exists);
  return cudaGetLastError();
}
cudaError_t aster_partial_aggregate(const uint64_t* gh, const double* v, const uint32_t* runs, uint32_t n, uint64_t* keys, double* sums, uint32_t* counts, double* mins, double* maxs, uint32_t mask, cudaStream_t s) {
  PartialAggregateKernel<<<(n + 255) / 256, 256, 0, s>>>(gh, v, runs, n, keys, sums, counts, mins, maxs, mask);
  return cudaGetLastError();
}
cudaError_t aster_radix_partition(const uint64_t* hashes, uint32_t n, uint32_t bits, uint32_t* partition, cudaStream_t s) {
  RadixPartitionKernel<<<(n + 255) / 256, 256, 0, s>>>(hashes, n, bits, partition);
  return cudaGetLastError();
}
}

}  // namespace aster::cuda
