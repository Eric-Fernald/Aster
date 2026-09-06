#pragma once
#include <cstdint>
#include <functional>

#include "aster/integration/plan_ir.hpp"
#include "aster/memory/bandwidth_probe.hpp"
#include "aster/memory/residency_map.hpp"
#include "aster/planner/physical_plan.hpp"

namespace aster::planner {

struct CostParams {
  double gpu_rows_per_sec = 5e9;
  double cpu_rows_per_sec = 2e8;
  double hbm_bytes_per_sec = 3e12;
  double jit_compile_seconds = 0.15;
  double equality_selectivity = 0.1;
  double range_selectivity = 0.33;
  double default_selectivity = 0.5;
  double like_selectivity = 0.2;
};

// Prices what the host optimizer cannot know: where the data is and what moving it costs.
class CostModel {
 public:
  CostModel(const memory::BandwidthTable& bw, const memory::ResidencyMap* residency, CostParams params = {});
  double Selectivity(const plan::Expr& pred) const;
  uint64_t EstimateRows(const plan::Rel& rel, const std::function<uint64_t(const plan::Rel&)>& source_rows) const;
  // Fills byte and time estimates for one pipeline given current residency.
  void EstimatePipeline(Pipeline& p, const std::vector<SegmentScan>& scans, bool kernel_cached) const;
  double TransferSeconds(memory::Tier from, memory::Tier to, uint64_t bytes) const { return bw_.SecondsFor(from, to, bytes); }
  const CostParams& params() const { return params_; }

 private:
  const memory::BandwidthTable& bw_;
  const memory::ResidencyMap* residency_;
  CostParams params_;
};

}  // namespace aster::planner
