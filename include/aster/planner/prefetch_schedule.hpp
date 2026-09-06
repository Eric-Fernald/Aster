#pragma once
#include <vector>

#include "aster/memory/page.hpp"
#include "aster/planner/physical_plan.hpp"

namespace aster::planner {

// Orders page fetches so the first tiles of each pipeline are in flight while kernels compile.
class PrefetchScheduler {
 public:
  explicit PrefetchScheduler(uint32_t depth) : depth_(depth) {}
  std::vector<memory::PageKey> Schedule(const PhysicalPlan& plan) const;
  // Pages the first `depth` tiles of a pipeline need, in access order.
  std::vector<memory::PageKey> InitialWindow(const Pipeline& p) const;

 private:
  uint32_t depth_;
};

}  // namespace aster::planner
