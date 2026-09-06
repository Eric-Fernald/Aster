#pragma once
#include <vector>

#include "aster/common/column.hpp"
#include "aster/exec/tile.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::exec {

// Phase 2 operator, CPU only. Whole partition frames: row_number, rank, dense_rank, lag, lead and
// the running aggregates sum/count/min/max/avg ordered by the window's ORDER BY.
class WindowOperator {
 public:
  WindowOperator(std::vector<plan::WindowFn> fns, Schema in_schema, Schema out_schema);
  Result<RecordBatchPtr> Execute(const RecordBatch& in);

 private:
  Result<Column> ComputeOne(const plan::WindowFn& fn, const RecordBatch& in, const std::vector<std::vector<RowIdx>>& partitions);
  std::vector<plan::WindowFn> fns_;
  Schema in_schema_, out_schema_;
};

}  // namespace aster::exec
