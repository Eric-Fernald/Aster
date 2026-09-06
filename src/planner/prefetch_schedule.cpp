#include "aster/planner/prefetch_schedule.hpp"

#include <set>

namespace aster::planner {

std::vector<memory::PageKey> PrefetchScheduler::InitialWindow(const Pipeline& p) const {
  // Pages are ordered segment by segment; the first `depth` segments cover the first tiles.
  std::vector<memory::PageKey> out;
  if (p.pages.empty()) return out;
  std::set<SegmentId> segs;
  for (const auto& k : p.pages) {
    if (!segs.count(k.segment_id)) {
      if (segs.size() >= depth_) break;
      segs.insert(k.segment_id);
    }
    out.push_back(k);
  }
  return out;
}

std::vector<memory::PageKey> PrefetchScheduler::Schedule(const PhysicalPlan& plan) const {
  std::vector<memory::PageKey> order;
  std::set<memory::PageKey> seen;
  auto push = [&](const memory::PageKey& k) { if (seen.insert(k).second) order.push_back(k); };
  // First the opening window of every GPU pipeline so kernels compile while data lands, then the rest in order.
  for (const auto& p : plan.pipelines)
    if (!p.cpu) for (const auto& k : InitialWindow(p)) push(k);
  for (const auto& p : plan.pipelines)
    if (!p.cpu) for (const auto& k : p.pages) push(k);
  return order;
}

}  // namespace aster::planner
