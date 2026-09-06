#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace aster {

// xxh64 style hash for buffers and keys.
uint64_t Hash64(const void* data, size_t len, uint64_t seed = 0);
inline uint64_t Hash64(std::string_view s, uint64_t seed = 0) { return Hash64(s.data(), s.size(), seed); }
inline uint64_t HashInt(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return x;
}
inline uint64_t HashCombine(uint64_t a, uint64_t b) { return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)); }

// CRC32C (Castagnoli), per chunk checksum verified during decode.
uint32_t Crc32c(const void* data, size_t len, uint32_t crc = 0);

}  // namespace aster
