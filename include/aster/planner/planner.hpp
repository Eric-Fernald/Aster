#pragma once
#include <functional>
#include <memory>

#include "aster/common/config.hpp"
#include "aster/integration/capability_registry.hpp"
#include "aster/integration/fallback_router.hpp"
#include "aster/memory/memory_manager.hpp"
#include "aster/planner/cost_model.hpp"
#include "aster/planner/physical_plan.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::planner {

struct PlannerContext {
  const EngineConfig* config = nullptr;
  memory::MemoryManager* memory = nullptr;
  storage::Catalog* catalog = nullptr;
  const integration::CapabilityRegistry* registry = nullptr;
  std::function<bool(const std::string& shape_hash)> kernel_cached;  // JIT cache probe
  std::function<uint64_t(const std::string& table)> delta_rows;      // rows waiting in the delta store
};

// The host optimizer chose join order and pushed predicates. This planner decides what it cannot:
// which segments to skip, where pipelines break, where each pipeline runs, and what to prefetch.
class Planner {
 public:
  explicit Planner(PlannerContext ctx);
  Result<PhysicalPlan> Plan(const plan::RelPtr& root);
  const integration::FallbackStats& last_fallback() const { return fallback_; }

 private:
  Status PruneScans(PhysicalPlan& plan);
  void EstimateAndPlace(PhysicalPlan& plan);
  uint64_t SourceRows(const plan::Rel& rel, const PhysicalPlan& plan) const;
  PlannerContext ctx_;
  integration::FallbackStats fallback_;
};

}  // namespace aster::planner
