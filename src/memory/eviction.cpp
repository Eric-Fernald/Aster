#include "aster/memory/eviction.hpp"

#include <algorithm>
#include <cmath>

namespace aster::memory {

double EvictionPolicy::Score(const PageDesc& d, uint64_t now_epoch) const {
  double age = double(now_epoch - std::min(now_epoch, d.last_access_epoch));
  double recency = cfg_.recency_weight / (1.0 + age);
  double frequency = cfg_.frequency_weight * std::log1p(double(d.access_count));
  double bonus = (d.dimension_hint && d.encoded_bytes <= cfg_.dimension_max_bytes) ? cfg_.dimension_bonus : 0.0;
  return recency + frequency + bonus;
}

std::vector<PageKey> EvictionPolicy::SelectVictims(ResidencyMap& map, Tier from, size_t bytes_needed,
                                                   uint64_t now_epoch) const {
  struct Cand { PageKey key; double score; size_t bytes; };
  std::vector<Cand> cands;
  map.ForEach([&](PageDesc& d) {
    if (d.current_tier() != from) return;
    if (d.pin_count > 0 || d.scheduled) return;
    cands.push_back({d.key(), Score(d, now_epoch), d.encoded_bytes});
  });
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });
  std::vector<PageKey> out;
  size_t freed = 0;
  for (const auto& c : cands) {
    if (freed >= bytes_needed) break;
    out.push_back(c.key);
    freed += c.bytes;
  }
  return out;
}

}  // namespace aster::memory
