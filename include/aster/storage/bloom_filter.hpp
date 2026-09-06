#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

#include "aster/common/column.hpp"

namespace aster::storage {

// Blocked bloom filter: each key touches one 256 bit block, GPU friendly and cache friendly.
class BloomFilter {
 public:
  BloomFilter() = default;
  BloomFilter(uint64_t expected_items, double fpp);
  static BloomFilter Build(const Column& col, double fpp = 0.01);

  void AddHash(uint64_t h);
  void Add(std::string_view s);
  void Add(int64_t v);
  bool MayContainHash(uint64_t h) const;
  bool MayContain(std::string_view s) const;
  bool MayContain(int64_t v) const;
  bool empty() const { return blocks_.empty(); }
  size_t bytes() const { return blocks_.size() * sizeof(uint32_t); }
  uint32_t num_blocks() const { return static_cast<uint32_t>(blocks_.size() / 8); }
  const std::vector<uint32_t>& raw() const { return blocks_; }

  void Serialize(std::vector<uint8_t>& out) const;
  static BloomFilter Deserialize(const uint8_t*& p, const uint8_t* end);

 private:
  std::vector<uint32_t> blocks_;  // 8 words per block
};

}  // namespace aster::storage
