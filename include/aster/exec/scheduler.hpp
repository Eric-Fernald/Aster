#pragma once
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "aster/exec/backend.hpp"
#include "aster/exec/fused/jit_compiler.hpp"
#include "aster/exec/operators/hash_aggregate.hpp"
#include "aster/exec/operators/hash_join.hpp"
#include "aster/exec/operators/sort.hpp"
#include "aster/exec/tile.hpp"
#include "aster/planner/physical_plan.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::exec {

// Result of one pipeline: either batches (output / sort input) or a breaker state consumed by later pipelines.
struct PipelineOutput {
  std::vector<RecordBatchPtr> batches;
  std::shared_ptr<HashJoinBuild> join_build;
  Schema schema;
  uint64_t rows_in = 0;
  uint64_t rows_out = 0;
  uint64_t bytes_scanned = 0;
  double kernel_ms = 0;
  double jit_ms = 0;
  uint32_t tiles = 0;
  bool fused = false;
};

struct SchedulerContext {
  ExecContext exec;
  OperatorBackend* backend = nullptr;
  fused::JitCompiler* jit = nullptr;
  storage::Catalog* catalog = nullptr;
  std::function<std::vector<RecordBatchPtr>(const std::string& table)> delta_batches;
  bool enable_fusion = true;
};

// Runs pipelines in dependency order. Tiles of one pipeline run in parallel across workers, each
// worker holding its own breaker state; states merge when the pipeline drains.
class TileScheduler {
 public:
  TileScheduler(const planner::PhysicalPlan& plan, SchedulerContext ctx);
  Result<PipelineOutput> Run();
  const std::map<int, PipelineOutput>& outputs() const { return outputs_; }

 private:
  Result<PipelineOutput> RunPipeline(const planner::Pipeline& p);
  Result<std::unique_ptr<TileSource>> MakeSource(const planner::Pipeline& p);
  Result<PipelineOutput> RunStreaming(const planner::Pipeline& p, TileSource& src);
  Status ApplyStreamingOps(const planner::Pipeline& p, size_t first_op, Tile& tile, std::vector<RecordBatchPtr>& emitted,
                           HashAggregateState* agg, LimitState* limit, std::vector<std::shared_ptr<HashJoinBuild>>& probes);
  const planner::PhysicalPlan& plan_;
  SchedulerContext ctx_;
  std::map<int, PipelineOutput> outputs_;
};

}  // namespace aster::exec
