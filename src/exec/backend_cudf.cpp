#include "aster/exec/backend.hpp"

#if ASTER_HAVE_CUDF
#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/device_buffer.hpp>

#include "aster/exec/expression_eval.hpp"

namespace aster::exec {

namespace {

cudf::data_type CudfType(const DataType& t) {
  switch (t.id) {
    case TypeId::Bool: return cudf::data_type{cudf::type_id::BOOL8};
    case TypeId::Int8: return cudf::data_type{cudf::type_id::INT8};
    case TypeId::Int16: return cudf::data_type{cudf::type_id::INT16};
    case TypeId::Int32: return cudf::data_type{cudf::type_id::INT32};
    case TypeId::Int64: return cudf::data_type{cudf::type_id::INT64};
    case TypeId::UInt8: return cudf::data_type{cudf::type_id::UINT8};
    case TypeId::UInt16: return cudf::data_type{cudf::type_id::UINT16};
    case TypeId::UInt32: return cudf::data_type{cudf::type_id::UINT32};
    case TypeId::UInt64: return cudf::data_type{cudf::type_id::UINT64};
    case TypeId::Float32: return cudf::data_type{cudf::type_id::FLOAT32};
    case TypeId::Float64: return cudf::data_type{cudf::type_id::FLOAT64};
    case TypeId::Date32: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    case TypeId::Timestamp: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
    case TypeId::Decimal64: return cudf::data_type{cudf::type_id::DECIMAL64, -static_cast<int32_t>(t.width)};
    case TypeId::String: return cudf::data_type{cudf::type_id::STRING};
    default: return cudf::data_type{cudf::type_id::EMPTY};
  }
}

// Views over Aster buffers already resident on device; dictionary columns are viewed as their int32 codes.
cudf::column_view ViewOf(const Column& c) {
  if (c.is_dictionary_encoded())
    return cudf::column_view(cudf::data_type{cudf::type_id::INT32}, static_cast<cudf::size_type>(c.length), c.values->data(),
                             c.validity ? reinterpret_cast<const cudf::bitmask_type*>(c.validity->data()) : nullptr,
                             static_cast<cudf::size_type>(c.null_count));
  if (c.type.id == TypeId::String) {
    cudf::column_view offsets(cudf::data_type{cudf::type_id::INT32}, static_cast<cudf::size_type>(c.length + 1), c.offsets->data(), nullptr, 0);
    return cudf::column_view(cudf::data_type{cudf::type_id::STRING}, static_cast<cudf::size_type>(c.length), c.values->data(),
                             c.validity ? reinterpret_cast<const cudf::bitmask_type*>(c.validity->data()) : nullptr,
                             static_cast<cudf::size_type>(c.null_count), 0, {offsets});
  }
  return cudf::column_view(CudfType(c.type), static_cast<cudf::size_type>(c.length), c.values->data(),
                           c.validity ? reinterpret_cast<const cudf::bitmask_type*>(c.validity->data()) : nullptr,
                           static_cast<cudf::size_type>(c.null_count));
}

cudf::table_view TableOf(const RecordBatch& b, std::vector<cudf::column_view>& hold) {
  hold.clear();
  for (const auto& c : b.columns) hold.push_back(ViewOf(c));
  return cudf::table_view(hold);
}

struct OwnedCol { std::unique_ptr<cudf::column> col; };
void ReleaseCudf(void*, size_t, void* ctx) { delete static_cast<OwnedCol*>(ctx); }

Column FromCudf(std::unique_ptr<cudf::column> col, const DataType& t, const Column* dict_source) {
  Column out;
  out.type = t;
  out.length = col->size();
  out.null_count = col->null_count();
  cudf::column_view v = col->view();
  if (v.type().id() == cudf::type_id::STRING) {
    auto offsets = v.child(0);
    out.offsets = std::make_shared<Buffer>(const_cast<void*>(offsets.head()), size_t(out.length + 1) * 4, MemorySpace::Device, nullptr, nullptr);
    out.values = std::make_shared<Buffer>(const_cast<void*>(static_cast<const void*>(v.head<char>())), 0, MemorySpace::Device, nullptr, nullptr);
  } else {
    out.values = std::make_shared<Buffer>(const_cast<void*>(v.head()), size_t(out.length) * cudf::size_of(v.type()), MemorySpace::Device, nullptr, nullptr);
  }
  if (v.nullable())
    out.validity = std::make_shared<Buffer>(const_cast<void*>(static_cast<const void*>(v.null_mask())), (out.length + 7) / 8, MemorySpace::Device, nullptr, nullptr);
  // The cudf column owns the memory; tie its lifetime to the values buffer.
  auto* hold = new OwnedCol{std::move(col)};
  out.values = std::make_shared<Buffer>(out.values->data(), out.values->size(), MemorySpace::Device, ReleaseCudf, hold);
  if (dict_source) { out.dictionary = dict_source->dictionary; out.dictionary_offsets = dict_source->dictionary_offsets; out.dictionary_id = dict_source->dictionary_id; }
  return out;
}

RecordBatchPtr FromCudf(std::unique_ptr<cudf::table> tbl, const Schema& schema, const std::vector<const Column*>& dict_sources) {
  auto out = std::make_shared<RecordBatch>();
  out->schema = schema;
  auto cols = tbl->release();
  for (size_t i = 0; i < cols.size(); ++i)
    out->columns.push_back(FromCudf(std::move(cols[i]), schema.fields[i].type, i < dict_sources.size() ? dict_sources[i] : nullptr));
  return out;
}

}  // namespace

Result<SelectionVector> CudfBackend::Filter(const plan::Expr& pred, const RecordBatch& in, const SelectionVector& sel, ExecContext& ctx) {
  // Predicate evaluation goes through the fused kernel path; libcudf handles the gather of survivors.
  return ExpressionEvaluator::Filter(pred, in, sel);
}

Result<RecordBatchPtr> CudfBackend::Project(const std::vector<plan::ExprPtr>& exprs, const Schema& out_schema, const RecordBatch& in, ExecContext& ctx) {
  CpuBackend cpu;
  return cpu.Project(exprs, out_schema, in, ctx);
}

Result<RecordBatchPtr> CudfBackend::HashJoin(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema, ExecContext& ctx) {
  std::vector<cudf::column_view> hb, hp;
  cudf::table_view bt = TableOf(build, hb), pt = TableOf(probe, hp);
  std::vector<cudf::size_type> bk(spec.right_keys.begin(), spec.right_keys.end());
  std::vector<cudf::size_type> pk(spec.left_keys.begin(), spec.left_keys.end());
  cudf::table_view build_keys = bt.select(bk), probe_keys = pt.select(pk);
  std::unique_ptr<cudf::table> result;
  try {
    switch (spec.type) {
      case plan::JoinType::Inner: {
        auto [pi, bi] = cudf::inner_join(probe_keys, build_keys);
        auto left = cudf::gather(pt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, pi->size(), pi->data(), nullptr, 0));
        auto right = cudf::gather(bt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, bi->size(), bi->data(), nullptr, 0));
        auto lc = left->release(); auto rc = right->release();
        for (auto& c : rc) lc.push_back(std::move(c));
        result = std::make_unique<cudf::table>(std::move(lc));
        break;
      }
      case plan::JoinType::Left: {
        auto [pi, bi] = cudf::left_join(probe_keys, build_keys);
        auto left = cudf::gather(pt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, pi->size(), pi->data(), nullptr, 0));
        auto right = cudf::gather(bt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, bi->size(), bi->data(), nullptr, 0), cudf::out_of_bounds_policy::NULLIFY);
        auto lc = left->release(); auto rc = right->release();
        for (auto& c : rc) lc.push_back(std::move(c));
        result = std::make_unique<cudf::table>(std::move(lc));
        break;
      }
      case plan::JoinType::Semi: {
        auto idx = cudf::left_semi_join(probe_keys, build_keys);
        result = cudf::gather(pt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, idx->size(), idx->data(), nullptr, 0));
        break;
      }
      case plan::JoinType::Anti: {
        auto idx = cudf::left_anti_join(probe_keys, build_keys);
        result = cudf::gather(pt, cudf::column_view(cudf::data_type{cudf::type_id::INT32}, idx->size(), idx->data(), nullptr, 0));
        break;
      }
      default: return Status::NotSupported("cudf join type");
    }
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cudf join: ") + e.what());
  }
  std::vector<const Column*> dicts;
  for (const auto& c : probe.columns) dicts.push_back(&c);
  for (const auto& c : build.columns) dicts.push_back(&c);
  return FromCudf(std::move(result), out_schema, dicts);
}

Result<RecordBatchPtr> CudfBackend::HashAggregate(const RecordBatch& in, const std::vector<plan::ExprPtr>& keys, const std::vector<plan::AggregateFn>& aggs, const Schema& out_schema, ExecContext& ctx) {
  // Keys and aggregate inputs must be plain column references for the libcudf path; the scheduler
  // projects complex expressions first.
  std::vector<cudf::column_view> hold;
  std::vector<cudf::column_view> key_views;
  for (const auto& k : keys) {
    if (!k->is_column()) return Status::NotSupported("cudf groupby needs column keys");
    key_views.push_back(ViewOf(in.columns[k->field_index]));
  }
  try {
    cudf::groupby::groupby gb(cudf::table_view(key_views));
    std::vector<cudf::groupby::aggregation_request> reqs;
    for (const auto& a : aggs) {
      cudf::groupby::aggregation_request r;
      int col = a.args.empty() ? (keys.empty() ? 0 : keys[0]->field_index) : a.args[0]->field_index;
      r.values = ViewOf(in.columns[col]);
      if (a.function == "sum") r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      else if (a.function == "count" || a.function == "count_star") r.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>(a.function == "count_star" ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE));
      else if (a.function == "min") r.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
      else if (a.function == "max") r.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
      else if (a.function == "avg") r.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
      else if (a.function == "count_distinct") r.aggregations.push_back(cudf::make_nunique_aggregation<cudf::groupby_aggregation>());
      else return Status::NotSupported("cudf aggregate " + a.function);
      reqs.push_back(std::move(r));
    }
    auto [gkeys, results] = gb.aggregate(reqs);
    std::vector<std::unique_ptr<cudf::column>> cols = gkeys->release();
    for (auto& r : results) for (auto& c : r.results) cols.push_back(std::move(c));
    std::vector<const Column*> dicts;
    for (const auto& k : keys) dicts.push_back(&in.columns[k->field_index]);
    return FromCudf(std::make_unique<cudf::table>(std::move(cols)), out_schema, dicts);
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cudf groupby: ") + e.what());
  }
}

Result<RecordBatchPtr> CudfBackend::Sort(const RecordBatch& in, const SortSpec& spec, ExecContext& ctx) {
  std::vector<cudf::column_view> hold;
  cudf::table_view t = TableOf(in, hold);
  std::vector<cudf::column_view> kv;
  std::vector<cudf::order> orders;
  std::vector<cudf::null_order> nulls;
  for (const auto& k : spec.keys) {
    if (!k.expr->is_column()) return Status::NotSupported("cudf sort needs column keys");
    kv.push_back(ViewOf(in.columns[k.expr->field_index]));
    orders.push_back(k.ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nulls.push_back(k.nulls_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
  }
  try {
    auto sorted = cudf::sort_by_key(t, cudf::table_view(kv), orders, nulls);
    if (spec.limit >= 0 && sorted->num_rows() > spec.limit) {
      auto sliced = cudf::slice(sorted->view(), {0, static_cast<cudf::size_type>(spec.limit)})[0];
      sorted = std::make_unique<cudf::table>(sliced);
    }
    std::vector<const Column*> dicts;
    for (const auto& c : in.columns) dicts.push_back(&c);
    return FromCudf(std::move(sorted), in.schema, dicts);
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cudf sort: ") + e.what());
  }
}

Result<RecordBatchPtr> CudfBackend::Concat(const std::vector<RecordBatchPtr>& batches, ExecContext& ctx) {
  if (batches.empty()) return std::make_shared<RecordBatch>();
  std::vector<std::vector<cudf::column_view>> holds(batches.size());
  std::vector<cudf::table_view> views;
  for (size_t i = 0; i < batches.size(); ++i) views.push_back(TableOf(*batches[i], holds[i]));
  try {
    auto t = cudf::concatenate(views);
    std::vector<const Column*> dicts;
    for (const auto& c : batches[0]->columns) dicts.push_back(&c);
    return FromCudf(std::move(t), batches[0]->schema, dicts);
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cudf concat: ") + e.what());
  }
}

}  // namespace aster::exec
#endif
