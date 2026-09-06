#include "aster/exec/operators/hash_aggregate.hpp"

#include <algorithm>
#include <cmath>

#include "aster/common/hash.hpp"
#include "aster/exec/expression_eval.hpp"

namespace aster::exec {

namespace {
inline bool IsFloatT(TypeId t) { return t == TypeId::Float32 || t == TypeId::Float64; }
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
    case TypeId::Decimal64: return double(c.Values<int64_t>()[i]) / std::pow(10.0, c.type.width);
    case TypeId::UInt64: return double(c.Values<uint64_t>()[i]);
    default: return double(c.Values<int64_t>()[i]);
  }
}
int64_t IntAt(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::UInt64: return static_cast<int64_t>(c.Values<uint64_t>()[i]);
    default: return c.Values<int64_t>()[i];
  }
}
}  // namespace

AggKind ParseAggKind(const std::string& fn) {
  if (fn == "sum") return AggKind::Sum;
  if (fn == "count") return AggKind::Count;
  if (fn == "count_star") return AggKind::CountStar;
  if (fn == "min") return AggKind::Min;
  if (fn == "max") return AggKind::Max;
  if (fn == "avg") return AggKind::Avg;
  if (fn == "count_distinct") return AggKind::CountDistinct;
  return AggKind::Count;
}

HashAggregateState::HashAggregateState(std::vector<plan::ExprPtr> keys, std::vector<plan::AggregateFn> aggs, Schema in_schema, Schema out_schema)
    : keys_(std::move(keys)), aggs_(std::move(aggs)), in_schema_(std::move(in_schema)), out_schema_(std::move(out_schema)) {
  for (const auto& a : aggs_) kinds_.push_back(ParseAggKind(a.function));
}

KeyValue HashAggregateState::KeyAt(const Column& c, int64_t row) {
  KeyValue k;
  if (!c.IsValid(row)) { k.is_null = true; return k; }
  if (c.is_dictionary_encoded() || IsStrT(c.type.id)) { k.kind = 2; k.s = std::string(c.GetString(row)); }
  else if (IsFloatT(c.type.id) || c.type.id == TypeId::Decimal64) { k.kind = 1; k.d = DoubleAt(c, row); }
  else { k.kind = 0; k.i = IntAt(c, row); }
  return k;
}

uint64_t HashAggregateState::HashKeys(const std::vector<KeyValue>& keys) {
  uint64_t h = 0x51ed270b27f1f4c3ULL;
  for (const auto& k : keys) {
    uint64_t v = k.is_null ? 0x7 : k.kind == 0 ? HashInt(static_cast<uint64_t>(k.i)) : k.kind == 1 ? Hash64(&k.d, 8) : Hash64(k.s);
    h = HashCombine(h, v);
  }
  return h;
}

size_t HashAggregateState::FindOrInsert(std::vector<KeyValue> keys, uint64_t h) {
  auto range = index_.equal_range(h);
  for (auto it = range.first; it != range.second; ++it)
    if (groups_[it->second].keys == keys) return it->second;
  Group g;
  g.keys = std::move(keys);
  g.states.resize(aggs_.size());
  groups_.push_back(std::move(g));
  index_.emplace(h, groups_.size() - 1);
  return groups_.size() - 1;
}

void HashAggregateState::Accumulate(AggState& st, AggKind kind, const Column* col, int64_t row, int64_t weight) {
  if (kind == AggKind::CountStar) { st.count += weight; st.seen = true; return; }
  if (!col || !col->IsValid(row)) return;
  st.seen = true;
  bool flt = IsFloatT(col->type.id) || col->type.id == TypeId::Decimal64;
  bool str = IsStrT(col->type.id) || col->is_dictionary_encoded();
  st.is_float |= flt;
  st.is_string |= str;
  switch (kind) {
    case AggKind::Count: st.count += weight; break;
    case AggKind::Sum: case AggKind::Avg:
      st.count += weight;
      if (flt) st.sum += DoubleAt(*col, row) * weight;
      else { st.isum += IntAt(*col, row) * weight; st.sum += double(IntAt(*col, row)) * weight; }
      break;
    case AggKind::Min: case AggKind::Max: {
      if (str) {
        std::string v(col->GetString(row));
        if (st.count == 0) { st.smin = st.smax = v; }
        else { if (v < st.smin) st.smin = v; if (v > st.smax) st.smax = v; }
      } else if (flt) {
        double v = DoubleAt(*col, row);
        if (st.count == 0) { st.dmin = st.dmax = v; } else { st.dmin = std::min(st.dmin, v); st.dmax = std::max(st.dmax, v); }
      } else {
        int64_t v = IntAt(*col, row);
        if (st.count == 0) { st.imin = st.imax = v; } else { st.imin = std::min(st.imin, v); st.imax = std::max(st.imax, v); }
      }
      st.count += weight;
      break;
    }
    case AggKind::CountDistinct: {
      if (!st.distinct) st.distinct = std::make_unique<std::unordered_set<uint64_t>>();
      uint64_t h = str ? Hash64(col->GetString(row)) : flt ? Hash64(&st.sum, 0) ^ HashInt(static_cast<uint64_t>(DoubleAt(*col, row) * 1e6)) : HashInt(static_cast<uint64_t>(IntAt(*col, row)));
      st.distinct->insert(h);
      break;
    }
    default: break;
  }
}

Status HashAggregateState::Update(const Tile& tile) {
  return Update(*tile.batch, tile.selection, tile.has_runs() ? &tile.run_lengths : nullptr);
}

Status HashAggregateState::Update(const RecordBatch& batch, const SelectionVector& sel, const std::vector<uint32_t>* run_lengths) {
  std::vector<Column> key_cols;
  for (const auto& k : keys_) { ASTER_ASSIGN_OR_RETURN(Column c, ExpressionEvaluator::Evaluate(*k, batch)); key_cols.push_back(std::move(c)); }
  std::vector<Column> agg_cols(aggs_.size());
  std::vector<bool> has_col(aggs_.size(), false);
  for (size_t a = 0; a < aggs_.size(); ++a) {
    if (aggs_[a].args.empty()) continue;
    ASTER_ASSIGN_OR_RETURN(agg_cols[a], ExpressionEvaluator::Evaluate(*aggs_[a].args[0], batch));
    has_col[a] = true;
  }
  int64_t n = sel.count(batch.num_rows());
  std::vector<KeyValue> keys(keys_.size());
  for (int64_t si = 0; si < n; ++si) {
    int64_t row = sel.all ? si : sel.rows[si];
    for (size_t k = 0; k < keys_.size(); ++k) keys[k] = KeyAt(key_cols[k], row);
    size_t g = FindOrInsert(keys, HashKeys(keys));
    int64_t weight = run_lengths ? (*run_lengths)[row] : 1;
    for (size_t a = 0; a < aggs_.size(); ++a) Accumulate(groups_[g].states[a], kinds_[a], has_col[a] ? &agg_cols[a] : nullptr, row, weight);
  }
  return Status::OK();
}

void HashAggregateState::MergeState(AggState& into, AggState& from, AggKind kind) {
  if (!from.seen) return;
  bool first = !into.seen;
  into.seen = true;
  into.is_float |= from.is_float;
  into.is_string |= from.is_string;
  switch (kind) {
    case AggKind::Count: case AggKind::CountStar: into.count += from.count; break;
    case AggKind::Sum: case AggKind::Avg: into.count += from.count; into.sum += from.sum; into.isum += from.isum; break;
    case AggKind::Min: case AggKind::Max:
      if (first) { into.smin = from.smin; into.smax = from.smax; into.dmin = from.dmin; into.dmax = from.dmax; into.imin = from.imin; into.imax = from.imax; }
      else {
        if (from.smin < into.smin) into.smin = from.smin;
        if (from.smax > into.smax) into.smax = from.smax;
        into.dmin = std::min(into.dmin, from.dmin); into.dmax = std::max(into.dmax, from.dmax);
        into.imin = std::min(into.imin, from.imin); into.imax = std::max(into.imax, from.imax);
      }
      into.count += from.count;
      break;
    case AggKind::CountDistinct:
      if (from.distinct) {
        if (!into.distinct) into.distinct = std::make_unique<std::unordered_set<uint64_t>>();
        into.distinct->insert(from.distinct->begin(), from.distinct->end());
      }
      break;
  }
}

Status HashAggregateState::Merge(HashAggregateState& other) {
  for (auto& g : other.groups_) {
    size_t idx = FindOrInsert(g.keys, HashKeys(g.keys));
    for (size_t a = 0; a < aggs_.size(); ++a) MergeState(groups_[idx].states[a], g.states[a], kinds_[a]);
  }
  other.groups_.clear();
  other.index_.clear();
  return Status::OK();
}

Result<RecordBatchPtr> HashAggregateState::Finalize() {
  auto out = std::make_shared<RecordBatch>();
  out->schema = out_schema_;
  size_t ng = groups_.size();
  if (ng == 0 && keys_.empty()) {
    // Empty input with no keys yields one row of empty aggregates.
    Group g; g.states.resize(aggs_.size());
    groups_.push_back(std::move(g));
    ng = 1;
  }
  for (size_t k = 0; k < keys_.size(); ++k) {
    const DataType& t = out_schema_.fields[k].type;
    std::vector<bool> valid(ng);
    for (size_t g = 0; g < ng; ++g) valid[g] = !groups_[g].keys[k].is_null;
    if (IsStrT(t.id)) {
      std::vector<std::string> v(ng);
      for (size_t g = 0; g < ng; ++g) v[g] = groups_[g].keys[k].s;
      Column c = MakeStringColumn(v);
      c.validity = MakeValidity(valid, &c.null_count);
      out->columns.push_back(std::move(c));
    } else if (IsFloatT(t.id)) {
      std::vector<double> v(ng);
      for (size_t g = 0; g < ng; ++g) v[g] = groups_[g].keys[k].d;
      Column c = MakeColumn<double>(TypeId::Float64, v);
      c.validity = MakeValidity(valid, &c.null_count);
      out->columns.push_back(std::move(c));
    } else {
      std::vector<int64_t> v(ng);
      for (size_t g = 0; g < ng; ++g) v[g] = groups_[g].keys[k].kind == 1 ? static_cast<int64_t>(groups_[g].keys[k].d) : groups_[g].keys[k].i;
      Column c;
      if (t.id == TypeId::Int32 || t.id == TypeId::Date32) { std::vector<int32_t> v32(ng); for (size_t g = 0; g < ng; ++g) v32[g] = static_cast<int32_t>(v[g]); c = MakeColumn<int32_t>(t.id, v32); }
      else if (t.id == TypeId::Int16) { std::vector<int16_t> v16(ng); for (size_t g = 0; g < ng; ++g) v16[g] = static_cast<int16_t>(v[g]); c = MakeColumn<int16_t>(t.id, v16); }
      else if (t.id == TypeId::Int8 || t.id == TypeId::Bool || t.id == TypeId::UInt8) { std::vector<uint8_t> v8(ng); for (size_t g = 0; g < ng; ++g) v8[g] = static_cast<uint8_t>(v[g]); c = MakeColumn<uint8_t>(t.id, v8); }
      else { c = MakeColumn<int64_t>(TypeId::Int64, v); c.type = t.id == TypeId::Null ? DataType::Of(TypeId::Int64) : t; }
      c.validity = MakeValidity(valid, &c.null_count);
      out->columns.push_back(std::move(c));
    }
  }
  for (size_t a = 0; a < aggs_.size(); ++a) {
    AggKind kind = kinds_[a];
    const DataType& t = out_schema_.fields[keys_.size() + a].type;
    std::vector<bool> valid(ng, true);
    bool any_float = false, any_string = false;
    for (size_t g = 0; g < ng; ++g) { any_float |= groups_[g].states[a].is_float; any_string |= groups_[g].states[a].is_string; }
    if (kind == AggKind::Count || kind == AggKind::CountStar || kind == AggKind::CountDistinct) {
      std::vector<int64_t> v(ng);
      for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; v[g] = kind == AggKind::CountDistinct ? (s.distinct ? static_cast<int64_t>(s.distinct->size()) : 0) : s.count; }
      out->columns.push_back(MakeColumn<int64_t>(TypeId::Int64, v));
    } else if (kind == AggKind::Avg || (kind == AggKind::Sum && (any_float || IsFloatT(t.id)))) {
      std::vector<double> v(ng);
      for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; valid[g] = s.seen && s.count > 0; v[g] = kind == AggKind::Avg ? (s.count ? s.sum / s.count : 0) : s.sum; }
      Column c = MakeColumn<double>(TypeId::Float64, v);
      c.validity = MakeValidity(valid, &c.null_count);
      out->columns.push_back(std::move(c));
    } else if (kind == AggKind::Sum) {
      std::vector<int64_t> v(ng);
      for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; valid[g] = s.seen; v[g] = s.isum; }
      Column c = MakeColumn<int64_t>(TypeId::Int64, v);
      if (t.id == TypeId::Decimal64) c.type = t;
      c.validity = MakeValidity(valid, &c.null_count);
      out->columns.push_back(std::move(c));
    } else {
      bool is_min = kind == AggKind::Min;
      if (any_string || IsStrT(t.id)) {
        std::vector<std::string> v(ng);
        for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; valid[g] = s.seen; v[g] = is_min ? s.smin : s.smax; }
        Column c = MakeStringColumn(v);
        c.validity = MakeValidity(valid, &c.null_count);
        out->columns.push_back(std::move(c));
      } else if (any_float || IsFloatT(t.id)) {
        std::vector<double> v(ng);
        for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; valid[g] = s.seen; v[g] = is_min ? s.dmin : s.dmax; }
        Column c = MakeColumn<double>(TypeId::Float64, v);
        c.validity = MakeValidity(valid, &c.null_count);
        out->columns.push_back(std::move(c));
      } else {
        std::vector<int64_t> v(ng);
        for (size_t g = 0; g < ng; ++g) { const AggState& s = groups_[g].states[a]; valid[g] = s.seen; v[g] = is_min ? s.imin : s.imax; }
        Column c;
        if (t.id == TypeId::Int32 || t.id == TypeId::Date32) { std::vector<int32_t> v32(ng); for (size_t g = 0; g < ng; ++g) v32[g] = static_cast<int32_t>(v[g]); c = MakeColumn<int32_t>(t.id, v32); }
        else { c = MakeColumn<int64_t>(TypeId::Int64, v); if (t.id != TypeId::Null && IsFixedWidth(t.id) && TypeByteWidth(t) == 8) c.type = t; }
        c.validity = MakeValidity(valid, &c.null_count);
        out->columns.push_back(std::move(c));
      }
    }
  }
  return out;
}

Result<RecordBatchPtr> HashAggregateState::Combine(std::vector<std::unique_ptr<HashAggregateState>>& partials) {
  if (partials.empty()) return Status::Invalid("no partial aggregates");
  for (size_t i = 1; i < partials.size(); ++i) ASTER_RETURN_NOT_OK(partials[0]->Merge(*partials[i]));
  return partials[0]->Finalize();
}

}  // namespace aster::exec
