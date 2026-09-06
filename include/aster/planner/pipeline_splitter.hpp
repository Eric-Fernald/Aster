#pragma once
#include <vector>

#include "aster/integration/plan_ir.hpp"
#include "aster/planner/physical_plan.hpp"

namespace aster::planner {

// Splits the relational tree at pipeline breakers (hash build, sort, aggregate finalize, exchange)
// and emits pipelines in dependency order. Each pipeline becomes one fused kernel candidate.
class PipelineSplitter {
 public:
  static std::vector<Pipeline> Split(const plan::RelPtr& root);
  static std::string ShapeHash(const Pipeline& p);
  static bool IsFusable(const Pipeline& p);
};

}  // namespace aster::planner
