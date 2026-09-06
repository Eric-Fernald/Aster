#include <algorithm>

#include "aster/exec/backend.hpp"
#include "aster/exec/expression_eval.hpp"
#include "aster/exec/operators/hash_aggregate.hpp"
#include "aster/exec/operators/hash_join.hpp"
#include "aster/exec/operators/sort.hpp"

namespace aster::exec {

Result<SelectionVector> CpuBackend::Filter(const plan::Expr& pred, const RecordBatch& in, const SelectionVector& sel, ExecContext&) {
  return ExpressionEvaluator::Filter(pred, in, sel);
}

Result<RecordBatchPtr> CpuBackend::Project(const std::vector<plan::ExprPtr>& exprs, const Schema& out_schema, const RecordBatch& in, ExecContext&) {
  auto out = std::make_shared<RecordBatch>();
  out->schema = out_schema;
  for (size_t i = 0; i < exprs.size(); ++i) {
    ASTER_ASSIGN_OR_RETURN(Column c, ExpressionEvaluator::Evaluate(*exprs[i], in));
    if (i < out_schema.fields.size() && out_schema.fields[i].type.id != TypeId::Null && !(c.type == out_schema.fields[i].type) &&
        !c.is_dictionary_encoded() && IsFixedWidth(out_schema.fields[i].type.id) && IsFixedWidth(c.type.id)) {
      ASTER_ASSIGN_OR_RETURN(c, ExpressionEvaluator::Cast(c, out_schema.fields[i].type));
    }
    out->columns.push_back(std::move(c));
  }
  return out;
}

Result<RecordBatchPtr> CpuBackend::HashJoin(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema, ExecContext& ctx) {
  HashJoinBuild hj(spec, build.schema, out_schema);
  ASTER_RETURN_NOT_OK(hj.Add(build));
  ASTER_RETURN_NOT_OK(hj.Finish(ctx));
  return hj.Probe(probe, *this, ctx);
}

Result<RecordBatchPtr> CpuBackend::HashAggregate(const RecordBatch& in, const std::vector<plan::ExprPtr>& keys, const std::vector<plan::AggregateFn>& aggs, const Schema& out_schema, ExecContext&) {
  HashAggregateState st(keys, aggs, in.schema, out_schema);
  ASTER_RETURN_NOT_OK(st.Update(in, SelectionVector::All(), nullptr));
  return st.Finalize();
}

Result<RecordBatchPtr> CpuBackend::Sort(const RecordBatch& in, const SortSpec& spec, ExecContext&) {
  ASTER_ASSIGN_OR_RETURN(auto idx, SortAccumulator::SortIndices(in, spec));
  if (spec.limit >= 0 && static_cast<int64_t>(idx.size()) > spec.limit) idx.resize(spec.limit);
  return TakeBatch(in, idx);
}

Result<RecordBatchPtr> CpuBackend::Concat(const std::vector<RecordBatchPtr>& batches, ExecContext&) {
  return ConcatBatches(batches);
}

std::unique_ptr<OperatorBackend> MakeBackend(hal::Backend device_backend, bool prefer_cudf) {
#if ASTER_HAVE_CUDF
  if (prefer_cudf && device_backend == hal::Backend::Cuda) return std::make_unique<CudfBackend>();
#else
  (void)device_backend; (void)prefer_cudf;
#endif
  return std::make_unique<CpuBackend>();
}

}  // namespace aster::exec
