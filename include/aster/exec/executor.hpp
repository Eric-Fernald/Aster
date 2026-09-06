#pragma once
#include <memory>
#include <vector>

#include "aster/common/config.hpp"
#include "aster/exec/backend.hpp"
#include "aster/exec/fused/jit_compiler.hpp"
#include "aster/exec/scheduler.hpp"
#include "aster/memory/memory_manager.hpp"
#include "aster/metrics/metrics.hpp"
#include "aster/planner/physical_plan.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::exec {

struct QueryResult {
  Schema schema;
  std::vector<RecordBatchPtr> batches;
  metrics::QueryMetrics metrics;
  int64_t num_rows() const { int64_t n = 0; for (const auto& b : batches) n += b->num_rows(); return n; }
  RecordBatchPtr Concat() const { return ConcatBatches(batches); }
};

struct ExecutorDeps {
  const EngineConfig* config = nullptr;
  memory::MemoryManager* memory = nullptr;
  storage::Catalog* catalog = nullptr;
  fused::JitCompiler* jit = nullptr;
  std::function<std::vector<RecordBatchPtr>(const std::string& table)> delta_batches;
  hal::DevicePtr device;
};

// Drives a physical plan to completion: prefetch, schedule tiles, materialize results, record metrics.
// Results land in pinned host memory unless the caller keeps them on device for DLPack export.
class Executor {
 public:
  explicit Executor(ExecutorDeps deps);
  Result<QueryResult> Execute(const planner::PhysicalPlan& plan, const std::string& query_id, bool keep_on_device = false);
  OperatorBackend& backend() { return *backend_; }

 private:
  ExecutorDeps deps_;
  std::unique_ptr<OperatorBackend> backend_;
  uint64_t next_query_id_ = 1;
};

}  // namespace aster::exec
