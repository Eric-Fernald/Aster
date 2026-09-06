#include "aster/common/hash.hpp"

#include <cstring>

namespace aster {

namespace {
constexpr uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P3 = 1609587929392839161ULL,
                   P4 = 9650029242287828579ULL, P5 = 2870177450012600261ULL;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t round64(uint64_t acc, uint64_t in) { acc += in * P2; acc = rotl(acc, 31); return acc * P1; }
inline uint64_t merge(uint64_t acc, uint64_t v) { v = round64(0, v); acc ^= v; return acc * P1 + P4; }
inline uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
}  // namespace

uint64_t Hash64(const void* data, size_t len, uint64_t seed) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  const uint8_t* end = p + len;
  uint64_t h;
  if (len >= 32) {
    uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
    const uint8_t* lim = end - 32;
    do {
      v1 = round64(v1, rd64(p)); v2 = round64(v2, rd64(p + 8));
      v3 = round64(v3, rd64(p + 16)); v4 = round64(v4, rd64(p + 24));
      p += 32;
    } while (p <= lim);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
  } else {
    h = seed + P5;
  }
  h += len;
  while (p + 8 <= end) { h ^= round64(0, rd64(p)); h = rotl(h, 27) * P1 + P4; p += 8; }
  if (p + 4 <= end) { h ^= uint64_t(rd32(p)) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
  while (p < end) { h ^= uint64_t(*p++) * P5; h = rotl(h, 11) * P1; }
  h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
  return h;
}

namespace {
struct CrcTable {
  uint32_t t[256];
  CrcTable() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
      t[i] = c;
    }
  }
};
}  // namespace

uint32_t Crc32c(const void* data, size_t len, uint32_t crc) {
  static const CrcTable tbl;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  for (size_t i = 0; i < len; ++i) crc = tbl.t[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
  return ~crc;
}

}  // namespace aster
