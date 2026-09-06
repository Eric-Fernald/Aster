#pragma once
#include <memory>
#include <vector>

#include "aster/exec/backend.hpp"
#include "aster/exec/tile.hpp"
#include "aster/memory/spill_controller.hpp"

namespace aster::exec {

// Collects tiles into sorted runs; runs spill to host/NVMe past the budget and merge at finalize.
// A limit turns this into top-k so only k rows are ever retained per run.
class SortAccumulator {
 public:
  SortAccumulator(SortSpec spec, Schema schema, size_t run_budget_bytes = size_t(2) << 30);
  Status Add(const Tile& tile, OperatorBackend& backend, ExecContext& ctx);
  Status Add(RecordBatchPtr batch, OperatorBackend& backend, ExecContext& ctx);
  Result<RecordBatchPtr> Finalize(OperatorBackend& backend, ExecContext& ctx);
  size_t num_runs() const { return runs_.size(); }
  static Result<std::vector<RowIdx>> SortIndices(const RecordBatch& b, const SortSpec& spec);
  static int CompareRows(const RecordBatch& a, int64_t ra, const RecordBatch& b, int64_t rb, const SortSpec& spec);

 private:
  Status FlushPending(OperatorBackend& backend, ExecContext& ctx);
  SortSpec spec_;
  Schema schema_;
  size_t budget_;
  std::vector<RecordBatchPtr> pending_;
  size_t pending_bytes_ = 0;
  std::vector<RecordBatchPtr> runs_;
};

struct LimitState {
  int64_t offset = 0;
  int64_t count = -1;
  int64_t seen = 0;
  int64_t emitted = 0;
  bool done() const { return count >= 0 && emitted >= count; }
  // Returns the rows of the tile that fall inside the window.
  SelectionVector Apply(const Tile& tile);
};

}  // namespace aster::exec
