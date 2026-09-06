#pragma once
#include <memory>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/exec/tile.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::exec {

struct JoinSpec {
  plan::JoinType type = plan::JoinType::Inner;
  std::vector<int> left_keys, right_keys;
};

struct SortSpec {
  std::vector<plan::SortKey> keys;
  int64_t limit = -1;
};

// Operator library boundary. The CPU backend is the always working reference; the cudf backend wraps
// libcudf. Both sit behind this interface so swapping is a config flag.
class OperatorBackend {
 public:
  virtual ~OperatorBackend() = default;
  virtual const char* name() const = 0;
  virtual Result<SelectionVector> Filter(const plan::Expr& pred, const RecordBatch& in, const SelectionVector& sel, ExecContext& ctx) = 0;
  virtual Result<RecordBatchPtr> Project(const std::vector<plan::ExprPtr>& exprs, const Schema& out_schema, const RecordBatch& in, ExecContext& ctx) = 0;
  virtual Result<RecordBatchPtr> HashJoin(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema, ExecContext& ctx) = 0;
  virtual Result<RecordBatchPtr> HashAggregate(const RecordBatch& in, const std::vector<plan::ExprPtr>& keys, const std::vector<plan::AggregateFn>& aggs, const Schema& out_schema, ExecContext& ctx) = 0;
  virtual Result<RecordBatchPtr> Sort(const RecordBatch& in, const SortSpec& spec, ExecContext& ctx) = 0;
  virtual Result<RecordBatchPtr> Concat(const std::vector<RecordBatchPtr>& batches, ExecContext& ctx) = 0;
};

class CpuBackend final : public OperatorBackend {
 public:
  const char* name() const override { return "cpu"; }
  Result<SelectionVector> Filter(const plan::Expr& pred, const RecordBatch& in, const SelectionVector& sel, ExecContext& ctx) override;
  Result<RecordBatchPtr> Project(const std::vector<plan::ExprPtr>& exprs, const Schema& out_schema, const RecordBatch& in, ExecContext& ctx) override;
  Result<RecordBatchPtr> HashJoin(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema, ExecContext& ctx) override;
  Result<RecordBatchPtr> HashAggregate(const RecordBatch& in, const std::vector<plan::ExprPtr>& keys, const std::vector<plan::AggregateFn>& aggs, const Schema& out_schema, ExecContext& ctx) override;
  Result<RecordBatchPtr> Sort(const RecordBatch& in, const SortSpec& spec, ExecContext& ctx) override;
  Result<RecordBatchPtr> Concat(const std::vector<RecordBatchPtr>& batches, ExecContext& ctx) override;
};

#if ASTER_HAVE_CUDF
class CudfBackend final : public OperatorBackend {
 public:
  const char* name() const override { return "cudf"; }
  Result<SelectionVector> Filter(const plan::Expr& pred, const RecordBatch& in, const SelectionVector& sel, ExecContext& ctx) override;
  Result<RecordBatchPtr> Project(const std::vector<plan::ExprPtr>& exprs, const Schema& out_schema, const RecordBatch& in, ExecContext& ctx) override;
  Result<RecordBatchPtr> HashJoin(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema, ExecContext& ctx) override;
  Result<RecordBatchPtr> HashAggregate(const RecordBatch& in, const std::vector<plan::ExprPtr>& keys, const std::vector<plan::AggregateFn>& aggs, const Schema& out_schema, ExecContext& ctx) override;
  Result<RecordBatchPtr> Sort(const RecordBatch& in, const SortSpec& spec, ExecContext& ctx) override;
  Result<RecordBatchPtr> Concat(const std::vector<RecordBatchPtr>& batches, ExecContext& ctx) override;
};
#endif

std::unique_ptr<OperatorBackend> MakeBackend(hal::Backend device_backend, bool prefer_cudf);

}  // namespace aster::exec
