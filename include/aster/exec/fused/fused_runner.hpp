#pragma once
#include <memory>
#include <vector>

#include "aster/exec/fused/jit_compiler.hpp"
#include "aster/exec/tile.hpp"
#include "aster/planner/physical_plan.hpp"

namespace aster::exec::fused {

struct FusedLaunchStats {
  uint32_t launches = 0;
  double kernel_ms = 0;
  uint64_t rows_in = 0;
  uint64_t rows_out = 0;
};

// Launches the compiled pipeline kernel over encoded pages: one thread block per tile, selection
// vector and partial aggregates in shared memory, output materialized only at the breaker.
class FusedPipelineRunner {
 public:
  FusedPipelineRunner(JitCompiler& jit, const planner::Pipeline& pipeline, std::vector<ColumnBinding> inputs, uint32_t tile_rows);
  static bool CanRun(const planner::Pipeline& p, hal::Backend backend);
  Status Prepare();
  // Runs the kernel over one tile of encoded device pages. Output buffers are device resident.
  Result<RecordBatchPtr> RunTile(const std::vector<memory::PageHandle>& pages, uint32_t num_rows, ExecContext& ctx);
  const KernelSpec& spec() const { return spec_; }
  const FusedLaunchStats& stats() const { return stats_; }

 private:
  JitCompiler& jit_;
  const planner::Pipeline& pipeline_;
  std::vector<ColumnBinding> inputs_;
  uint32_t tile_rows_;
  KernelSpec spec_;
  std::shared_ptr<CompiledKernel> kernel_;
  FusedLaunchStats stats_;
};

}  // namespace aster::exec::fused
