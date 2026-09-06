#include "aster/exec/operators/sort.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

#include "aster/exec/expression_eval.hpp"

namespace aster::exec {

namespace {
inline bool IsStrT(TypeId t) { return t == TypeId::String || t == TypeId::Binary; }
double DoubleAt(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::Float32: return c.Values<float>()[i];
    case TypeId::Float64: return c.Values<double>()[i];
    case TypeId::UInt64: return double(c.Values<uint64_t>()[i]);
    default: return double(c.Values<int64_t>()[i]);
  }
}
int CompareCell(const Column& a, int64_t ra, const Column& b, int64_t rb, const plan::SortKey& k) {
  bool va = a.IsValid(ra), vb = b.IsValid(rb);
  if (!va || !vb) {
    if (va == vb) return 0;
    int null_first = k.nulls_first ? -1 : 1;
    return !va ? null_first : -null_first;
  }
  int c;
  if (a.is_dictionary_encoded() || IsStrT(a.type.id)) c = a.GetString(ra).compare(b.GetString(rb));
  else { double x = DoubleAt(a, ra), y = DoubleAt(b, rb); c = x < y ? -1 : x > y ? 1 : 0; }
  return k.ascending ? c : -c;
}
}  // namespace

int SortAccumulator::CompareRows(const RecordBatch& a, int64_t ra, const RecordBatch& b, int64_t rb, const SortSpec& spec) {
  for (const auto& k : spec.keys) {
    int idx = k.expr->field_index;
    int c = CompareCell(a.columns[idx], ra, b.columns[idx], rb, k);
    if (c) return c;
  }
  return 0;
}

Result<std::vector<RowIdx>> SortAccumulator::SortIndices(const RecordBatch& b, const SortSpec& spec) {
  // Non column keys are evaluated into a temporary key table.
  RecordBatch keyed;
  SortSpec local = spec;
  for (size_t i = 0; i < spec.keys.size(); ++i) {
    ASTER_ASSIGN_OR_RETURN(Column c, ExpressionEvaluator::Evaluate(*spec.keys[i].expr, b));
    keyed.columns.push_back(std::move(c));
    local.keys[i].expr = plan::Col(static_cast<int>(i));
  }
  std::vector<RowIdx> idx(b.num_rows());
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](RowIdx x, RowIdx y) { int c = CompareRows(keyed, x, keyed, y, local); return c ? c < 0 : x < y; };
  if (spec.limit >= 0 && spec.limit < b.num_rows()) {
    std::partial_sort(idx.begin(), idx.begin() + spec.limit, idx.end(), cmp);
    idx.resize(spec.limit);
  } else {
    std::stable_sort(idx.begin(), idx.end(), cmp);
  }
  return idx;
}

SortAccumulator::SortAccumulator(SortSpec spec, Schema schema, size_t run_budget_bytes)
    : spec_(std::move(spec)), schema_(std::move(schema)), budget_(run_budget_bytes) {}

Status SortAccumulator::Add(const Tile& tile, OperatorBackend& backend, ExecContext& ctx) {
  return Add(tile.Materialize(), backend, ctx);
}

Status SortAccumulator::Add(RecordBatchPtr batch, OperatorBackend& backend, ExecContext& ctx) {
  if (!batch || batch->num_rows() == 0) return Status::OK();
  pending_bytes_ += batch->nbytes();
  pending_.push_back(std::move(batch));
  if (pending_bytes_ >= budget_) return FlushPending(backend, ctx);
  return Status::OK();
}

Status SortAccumulator::FlushPending(OperatorBackend& backend, ExecContext& ctx) {
  if (pending_.empty()) return Status::OK();
  ASTER_ASSIGN_OR_RETURN(auto merged, backend.Concat(pending_, ctx));
  ASTER_ASSIGN_OR_RETURN(auto run, backend.Sort(*merged, spec_, ctx));
  runs_.push_back(run);
  pending_.clear();
  pending_bytes_ = 0;
  return Status::OK();
}

Result<RecordBatchPtr> SortAccumulator::Finalize(OperatorBackend& backend, ExecContext& ctx) {
  ASTER_RETURN_NOT_OK(FlushPending(backend, ctx));
  if (runs_.empty()) { auto e = std::make_shared<RecordBatch>(); e->schema = schema_; for (const auto& f : schema_.fields) e->columns.push_back(Column::Empty(f.type)); return e; }
  if (runs_.size() == 1) return runs_[0];
  // k-way merge of sorted runs; each run's key columns are column refs after Sort.
  std::vector<RecordBatch> keyed(runs_.size());
  SortSpec local = spec_;
  for (size_t r = 0; r < runs_.size(); ++r)
    for (size_t i = 0; i < spec_.keys.size(); ++i) {
      ASTER_ASSIGN_OR_RETURN(Column c, ExpressionEvaluator::Evaluate(*spec_.keys[i].expr, *runs_[r]));
      keyed[r].columns.push_back(std::move(c));
      local.keys[i].expr = plan::Col(static_cast<int>(i));
    }
  struct Head { size_t run; int64_t row; };
  auto greater = [&](const Head& a, const Head& b) {
    int c = CompareRows(keyed[a.run], a.row, keyed[b.run], b.row, local);
    return c ? c > 0 : a.run > b.run;
  };
  std::priority_queue<Head, std::vector<Head>, decltype(greater)> pq(greater);
  for (size_t r = 0; r < runs_.size(); ++r) if (runs_[r]->num_rows()) pq.push({r, 0});
  std::vector<std::vector<RowIdx>> take(runs_.size());
  std::vector<std::pair<size_t, RowIdx>> order;
  int64_t emitted = 0;
  while (!pq.empty() && (spec_.limit < 0 || emitted < spec_.limit)) {
    Head h = pq.top(); pq.pop();
    order.push_back({h.run, static_cast<RowIdx>(h.row)});
    ++emitted;
    if (h.row + 1 < runs_[h.run]->num_rows()) pq.push({h.run, h.row + 1});
  }
  // Materialize in merge order by gathering one row at a time through per run takes.
  std::vector<RecordBatchPtr> pieces;
  size_t i = 0;
  while (i < order.size()) {
    size_t r = order[i].first;
    std::vector<RowIdx> rows;
    while (i < order.size() && order[i].first == r) rows.push_back(order[i++].second);
    pieces.push_back(TakeBatch(*runs_[r], rows));
  }
  auto out = ConcatBatches(pieces);
  out->schema = schema_;
  return out;
}

SelectionVector LimitState::Apply(const Tile& tile) {
  int64_t n = tile.num_rows();
  std::vector<RowIdx> rows;
  for (int64_t i = 0; i < n; ++i) {
    RowIdx r = tile.selection.all ? static_cast<RowIdx>(i) : tile.selection.rows[i];
    if (seen++ < offset) continue;
    if (count >= 0 && emitted >= count) break;
    rows.push_back(r);
    ++emitted;
  }
  return SelectionVector::Of(std::move(rows));
}

}  // namespace aster::exec
