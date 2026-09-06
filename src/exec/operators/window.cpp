#include "aster/exec/operators/window.hpp"

#include <algorithm>
#include <map>
#include <numeric>

#include "aster/exec/expression_eval.hpp"
#include "aster/exec/operators/sort.hpp"

namespace aster::exec {

namespace {
double DoubleAt(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::Float32: return c.Values<float>()[i];
    case TypeId::Float64: return c.Values<double>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::Int8: case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    default: return double(c.Values<int64_t>()[i]);
  }
}
}  // namespace

WindowOperator::WindowOperator(std::vector<plan::WindowFn> fns, Schema in_schema, Schema out_schema)
    : fns_(std::move(fns)), in_schema_(std::move(in_schema)), out_schema_(std::move(out_schema)) {}

Result<RecordBatchPtr> WindowOperator::Execute(const RecordBatch& in) {
  auto out = std::make_shared<RecordBatch>(in);
  out->schema = out_schema_;
  for (const auto& fn : fns_) {
    // Partition rows by the partition keys, then order within each partition.
    std::map<std::string, std::vector<RowIdx>> parts;
    std::vector<Column> pcols;
    for (const auto& p : fn.partition_by) { ASTER_ASSIGN_OR_RETURN(Column c, ExpressionEvaluator::Evaluate(*p, in)); pcols.push_back(std::move(c)); }
    for (int64_t i = 0; i < in.num_rows(); ++i) {
      std::string key;
      for (const auto& c : pcols) {
        if (!c.IsValid(i)) key += "\x01null\x02";
        else if (c.type.id == TypeId::String || c.is_dictionary_encoded()) key += std::string(c.GetString(i)) + "\x02";
        else key += std::to_string(DoubleAt(c, i)) + "\x02";
      }
      parts[key].push_back(static_cast<RowIdx>(i));
    }
    std::vector<std::vector<RowIdx>> partitions;
    for (auto& [k, rows] : parts) {
      if (!fn.order_by.empty()) {
        RecordBatchPtr sub = TakeBatch(in, rows);
        SortSpec spec; spec.keys = fn.order_by;
        ASTER_ASSIGN_OR_RETURN(auto idx, SortAccumulator::SortIndices(*sub, spec));
        std::vector<RowIdx> ordered(rows.size());
        for (size_t i = 0; i < idx.size(); ++i) ordered[i] = rows[idx[i]];
        rows = ordered;
      }
      partitions.push_back(rows);
    }
    ASTER_ASSIGN_OR_RETURN(Column c, ComputeOne(fn, in, partitions));
    out->columns.push_back(std::move(c));
  }
  return out;
}

Result<Column> WindowOperator::ComputeOne(const plan::WindowFn& fn, const RecordBatch& in, const std::vector<std::vector<RowIdx>>& partitions) {
  int64_t n = in.num_rows();
  Column arg;
  bool has_arg = !fn.args.empty();
  if (has_arg) { ASTER_ASSIGN_OR_RETURN(arg, ExpressionEvaluator::Evaluate(*fn.args[0], in)); }
  Column order_col;
  if (!fn.order_by.empty()) { ASTER_ASSIGN_OR_RETURN(order_col, ExpressionEvaluator::Evaluate(*fn.order_by[0].expr, in)); }
  const std::string& f = fn.function;
  std::vector<int64_t> iv(n, 0);
  std::vector<double> dv(n, 0);
  std::vector<bool> valid(n, true);
  bool is_double = f == "avg" || f == "sum" || f == "min" || f == "max";
  if (f == "sum" && has_arg && IsIntegral(arg.type.id)) is_double = false;
  for (const auto& rows : partitions) {
    int64_t rank = 0, dense = 0;
    double run_sum = 0; int64_t run_count = 0; double run_min = 0, run_max = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      RowIdx r = rows[i];
      if (f == "row_number") { iv[r] = static_cast<int64_t>(i + 1); continue; }
      bool tie = i > 0 && !fn.order_by.empty() && order_col.IsValid(r) && order_col.IsValid(rows[i - 1]) && DoubleAt(order_col, r) == DoubleAt(order_col, rows[i - 1]);
      if (f == "rank") { if (!tie) rank = static_cast<int64_t>(i + 1); iv[r] = rank; continue; }
      if (f == "dense_rank") { if (!tie) ++dense; iv[r] = dense; continue; }
      if (f == "lag" || f == "lead") {
        int64_t off = fn.args.size() > 1 && fn.args[1]->is_literal() ? std::get<int64_t>(fn.args[1]->literal) : 1;
        int64_t j = f == "lag" ? static_cast<int64_t>(i) - off : static_cast<int64_t>(i) + off;
        if (j < 0 || j >= static_cast<int64_t>(rows.size()) || !arg.IsValid(rows[j])) { valid[r] = false; continue; }
        dv[r] = DoubleAt(arg, rows[j]); iv[r] = static_cast<int64_t>(dv[r]);
        continue;
      }
      // Running aggregates over the ordered partition (whole partition when there is no ORDER BY).
      if (has_arg && arg.IsValid(r)) {
        double v = DoubleAt(arg, r);
        if (run_count == 0) { run_min = run_max = v; } else { run_min = std::min(run_min, v); run_max = std::max(run_max, v); }
        run_sum += v; ++run_count;
      }
      bool whole = fn.order_by.empty();
      if (!whole) {
        if (f == "count") iv[r] = run_count;
        else if (f == "sum") { dv[r] = run_sum; iv[r] = static_cast<int64_t>(run_sum); }
        else if (f == "avg") dv[r] = run_count ? run_sum / run_count : 0;
        else if (f == "min") dv[r] = run_min;
        else if (f == "max") dv[r] = run_max;
        valid[r] = run_count > 0 || f == "count";
      }
    }
    if (fn.order_by.empty() && (f == "sum" || f == "avg" || f == "min" || f == "max" || f == "count")) {
      for (RowIdx r : rows) {
        if (f == "count") iv[r] = run_count;
        else if (f == "sum") { dv[r] = run_sum; iv[r] = static_cast<int64_t>(run_sum); }
        else if (f == "avg") dv[r] = run_count ? run_sum / run_count : 0;
        else if (f == "min") dv[r] = run_min;
        else dv[r] = run_max;
        valid[r] = run_count > 0 || f == "count";
      }
    }
  }
  bool any_null = std::find(valid.begin(), valid.end(), false) != valid.end();
  if (is_double || ((f == "lag" || f == "lead") && has_arg && (arg.type.id == TypeId::Float64 || arg.type.id == TypeId::Float32))) {
    Column c = MakeColumn<double>(TypeId::Float64, dv);
    if (any_null) c.validity = MakeValidity(valid, &c.null_count);
    return c;
  }
  Column c = MakeColumn<int64_t>(TypeId::Int64, iv);
  if (any_null) c.validity = MakeValidity(valid, &c.null_count);
  return c;
}

}  // namespace aster::exec
