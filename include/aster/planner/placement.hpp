#pragma once
#include "aster/memory/memory_manager.hpp"
#include "aster/planner/cost_model.hpp"
#include "aster/planner/physical_plan.hpp"

namespace aster::planner {

// Chooses per pipeline whether inputs are fully resident, streamed from host, streamed from NVMe,
// or executed on CPU. Reads the residency map and the HBM budget; never assumes.
class PlacementPolicy {
 public:
  PlacementPolicy(memory::MemoryManager* mm, const CostModel& cost) : mm_(mm), cost_(cost) {}
  void Assign(PhysicalPlan& plan) const;

 private:
  PlacementMode Choose(const Pipeline& p, size_t hbm_available) const;
  memory::MemoryManager* mm_;
  const CostModel& cost_;
};

}  // namespace aster::planner
