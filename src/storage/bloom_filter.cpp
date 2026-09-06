#include "aster/storage/bloom_filter.hpp"

#include <cmath>

#include "aster/common/hash.hpp"
#include "internal.hpp"

namespace aster::storage {

using namespace internal;

namespace {
constexpr uint32_t kSalts[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};
inline void BlockMask(uint32_t key, uint32_t out[8]) {
  for (int i = 0; i < 8; ++i) out[i] = 1u << ((key * kSalts[i]) >> 27);
}
}  // namespace

BloomFilter::BloomFilter(uint64_t expected_items, double fpp) {
  // ~ -n ln p / (ln 2)^2 bits, rounded to whole 256 bit blocks, at least one block.
  double bits = -double(expected_items ? expected_items : 1) * std::log(fpp) / (std::log(2.0) * std::log(2.0));
  uint64_t blocks = static_cast<uint64_t>(std::ceil(bits / 256.0));
  if (blocks == 0) blocks = 1;
  blocks_.assign(blocks * 8, 0);
}

BloomFilter BloomFilter::Build(const Column& col, double fpp) {
  BloomFilter b(static_cast<uint64_t>(col.length), fpp);
  const TypeId t = col.type.id;
  for (int64_t i = 0; i < col.length; ++i) {
    if (!col.IsValid(i)) continue;
    if (t == TypeId::String || t == TypeId::Binary) b.Add(col.GetString(i));
    else if (IsIntegerLike(t) || t == TypeId::Bool) b.Add(ReadI64(col, i));
    else { uint64_t bits; std::memcpy(&bits, col.values->data() + i * TypeByteWidth(col.type), std::min<size_t>(8, TypeByteWidth(col.type))); b.AddHash(HashInt(bits)); }
  }
  return b;
}

void BloomFilter::AddHash(uint64_t h) {
  if (blocks_.empty()) return;
  uint32_t nb = num_blocks();
  uint32_t block = static_cast<uint32_t>((h >> 32) % nb);
  uint32_t mask[8];
  BlockMask(static_cast<uint32_t>(h), mask);
  uint32_t* w = blocks_.data() + size_t(block) * 8;
  for (int i = 0; i < 8; ++i) w[i] |= mask[i];
}

bool BloomFilter::MayContainHash(uint64_t h) const {
  if (blocks_.empty()) return true;
  uint32_t nb = num_blocks();
  uint32_t block = static_cast<uint32_t>((h >> 32) % nb);
  uint32_t mask[8];
  BlockMask(static_cast<uint32_t>(h), mask);
  const uint32_t* w = blocks_.data() + size_t(block) * 8;
  for (int i = 0; i < 8; ++i)
    if ((w[i] & mask[i]) != mask[i]) return false;
  return true;
}

void BloomFilter::Add(std::string_view s) { AddHash(Hash64(s)); }
void BloomFilter::Add(int64_t v) { AddHash(HashInt(static_cast<uint64_t>(v))); }
bool BloomFilter::MayContain(std::string_view s) const { return MayContainHash(Hash64(s)); }
bool BloomFilter::MayContain(int64_t v) const { return MayContainHash(HashInt(static_cast<uint64_t>(v))); }

void BloomFilter::Serialize(std::vector<uint8_t>& out) const {
  PutVal<uint32_t>(out, static_cast<uint32_t>(blocks_.size()));
  Put(out, blocks_.data(), blocks_.size() * sizeof(uint32_t));
}

BloomFilter BloomFilter::Deserialize(const uint8_t*& p, const uint8_t*) {
  BloomFilter b;
  uint32_t n = GetVal<uint32_t>(p);
  b.blocks_.resize(n);
  std::memcpy(b.blocks_.data(), p, n * sizeof(uint32_t));
  p += n * sizeof(uint32_t);
  return b;
}

}  // namespace aster::storage
