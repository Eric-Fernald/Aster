#pragma once
#include <vector>

#include "aster/memory/page.hpp"
#include "aster/memory/residency_map.hpp"

namespace aster::memory {

struct EvictionConfig {
  double frequency_weight = 1.0;
  double recency_weight = 2.0;
  double dimension_bonus = 8.0;      // small dimension tables stay resident
  size_t dimension_max_bytes = size_t(256) << 20;
};

// Query aware policy: scheduled and pinned pages are never candidates; the rest are ranked by
// frequency and recency with a bias toward small dimension tables.
class EvictionPolicy {
 public:
  explicit EvictionPolicy(EvictionConfig cfg = {}) : cfg_(cfg) {}
  double Score(const PageDesc& d, uint64_t now_epoch) const;
  std::vector<PageKey> SelectVictims(ResidencyMap& map, Tier from, size_t bytes_needed, uint64_t now_epoch) const;

 private:
  EvictionConfig cfg_;
};

}  // namespace aster::memory
