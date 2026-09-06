#include "aster/planner/segment_pruner.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace aster::planner {

using namespace plan;
using storage::ColumnChunkMeta;

namespace {

const ColumnChunkMeta* ChunkFor(const Expr& col, const storage::SegmentMeta& seg, const Schema& read_schema,
                                const std::vector<int>& projection) {
  if (!col.is_column()) return nullptr;
  int idx = col.field_index;
  if (idx < 0 || idx >= static_cast<int>(projection.size())) return nullptr;
  const std::string& name = read_schema.fields.at(projection[idx]).name;
  int ci = seg.ColumnIndex(name);
  return ci < 0 ? nullptr : &seg.columns[ci];
}

bool LiteralI64(const Expr& e, int64_t* out) {
  if (!e.is_literal()) return false;
  if (auto i = std::get_if<int64_t>(&e.literal)) { *out = *i; return true; }
  if (auto b = std::get_if<bool>(&e.literal)) { *out = *b; return true; }
  return false;
}
bool LiteralF64(const Expr& e, double* out) {
  if (!e.is_literal()) return false;
  if (auto d = std::get_if<double>(&e.literal)) { *out = *d; return true; }
  int64_t i;
  if (LiteralI64(e, &i)) { *out = double(i); return true; }
  return false;
}
bool LiteralStr(const Expr& e, std::string* out) {
  if (!e.is_literal()) return false;
  if (auto s = std::get_if<std::string>(&e.literal)) { *out = *s; return true; }
  return false;
}

std::string Flip(const std::string& op) {
  if (op == "lt") return "gt";
  if (op == "lte") return "gte";
  if (op == "gt") return "lt";
  if (op == "gte") return "lte";
  return op;
}

bool Compare(const ColumnChunkMeta& c, const std::string& op, const Expr& lit) {
  const storage::ZoneMap& z = c.zone;
  if (!z.has_bounds) return true;
  if (c.zone.null_count == c.zone.row_count) return false;
  const TypeId t = c.type.id;
  if (t == TypeId::String) {
    std::string s;
    if (!LiteralStr(lit, &s)) return true;
    if (op == "equal") return z.MayContainStr(s) && (c.bloom.empty() || c.bloom.MayContain(s));
    if (op == "lt") return z.min_str < s;
    if (op == "lte") return z.min_str <= s;
    if (op == "gt") return z.max_str.size() == 64 || z.max_str > s;
    if (op == "gte") return z.max_str.size() == 64 || z.max_str >= s;
    if (op == "starts_with") return z.MayOverlapStr(s, s + "\xff");
    return true;
  }
  if (IsIntegerLike(t) || t == TypeId::Bool) {
    int64_t v;
    if (!LiteralI64(lit, &v)) { double d; if (!LiteralF64(lit, &d)) return true; v = static_cast<int64_t>(std::floor(d)); }
    if (op == "equal") return z.MayContainI64(v) && (c.bloom.empty() || c.bloom.MayContain(v));
    if (op == "lt") return z.min_i64 < v;
    if (op == "lte") return z.min_i64 <= v;
    if (op == "gt") return z.max_i64 > v;
    if (op == "gte") return z.max_i64 >= v;
    return true;
  }
  if (t == TypeId::Float32 || t == TypeId::Float64) {
    double v;
    if (!LiteralF64(lit, &v)) return true;
    if (op == "equal") return z.MayContainF64(v);
    if (op == "lt") return z.min_f64 < v;
    if (op == "lte") return z.min_f64 <= v;
    if (op == "gt") return z.max_f64 > v;
    if (op == "gte") return z.max_f64 >= v;
    return true;
  }
  return true;
}

}  // namespace

bool SegmentPruner::MayMatch(const Expr& pred, const storage::SegmentMeta& seg, const Schema& read_schema,
                             const std::vector<int>& projection) {
  if (pred.kind == ExprKind::Literal) {
    if (auto b = std::get_if<bool>(&pred.literal)) return *b;
    return true;
  }
  if (pred.kind != ExprKind::Call) return true;
  const std::string& f = pred.function;
  if (f == "and") {
    for (const auto& a : pred.args) if (!MayMatch(*a, seg, read_schema, projection)) return false;
    return true;
  }
  if (f == "or") {
    for (const auto& a : pred.args) if (MayMatch(*a, seg, read_schema, projection)) return true;
    return pred.args.empty();
  }
  if (f == "is_null" && pred.args.size() == 1) {
    const ColumnChunkMeta* c = ChunkFor(*pred.args[0], seg, read_schema, projection);
    return !c || c->null_count > 0;
  }
  if (f == "is_not_null" && pred.args.size() == 1) {
    const ColumnChunkMeta* c = ChunkFor(*pred.args[0], seg, read_schema, projection);
    return !c || c->null_count < c->num_rows;
  }
  if ((f == "equal" || f == "lt" || f == "lte" || f == "gt" || f == "gte" || f == "starts_with") && pred.args.size() == 2) {
    const Expr& a = *pred.args[0];
    const Expr& b = *pred.args[1];
    if (a.is_column() && b.is_literal()) { const ColumnChunkMeta* c = ChunkFor(a, seg, read_schema, projection); return !c || Compare(*c, f, b); }
    if (b.is_column() && a.is_literal()) { const ColumnChunkMeta* c = ChunkFor(b, seg, read_schema, projection); return !c || Compare(*c, Flip(f), a); }
    return true;
  }
  if (f == "between" && pred.args.size() == 3 && pred.args[0]->is_column()) {
    const ColumnChunkMeta* c = ChunkFor(*pred.args[0], seg, read_schema, projection);
    return !c || (Compare(*c, "gte", *pred.args[1]) && Compare(*c, "lte", *pred.args[2]));
  }
  if (f == "in" && pred.args.size() >= 2 && pred.args[0]->is_column()) {
    const ColumnChunkMeta* c = ChunkFor(*pred.args[0], seg, read_schema, projection);
    if (!c) return true;
    for (size_t i = 1; i < pred.args.size(); ++i) if (Compare(*c, "equal", *pred.args[i])) return true;
    return false;
  }
  return true;
}

PruneResult SegmentPruner::Prune(const std::vector<std::shared_ptr<storage::SegmentMeta>>& segments, const Expr* pred,
                                 const Schema& read_schema, const std::vector<int>& projection) {
  PruneResult r;
  for (const auto& s : segments) {
    if (pred && !MayMatch(*pred, *s, read_schema, projection)) { ++r.pruned_segments; r.pruned_rows += s->num_rows; continue; }
    r.kept.push_back(s);
    r.kept_rows += s->num_rows;
  }
  return r;
}

ExprPtr SegmentPruner::CollectPredicates(const RelPtr& read, const RelPtr& root) {
  std::vector<ExprPtr> conj;
  if (read->pushed_filter) conj.push_back(read->pushed_filter);
  // Walk down from root looking for Filter nodes whose only descendant chain to `read` is Filter/Project(identity).
  std::function<bool(const RelPtr&)> descend = [&](const RelPtr& r) -> bool {
    if (r == read) return true;
    if (r->inputs.size() != 1) return false;
    if (r->kind == RelKind::Filter) {
      if (descend(r->inputs[0])) { conj.push_back(r->predicate); return true; }
      return false;
    }
    if (r->kind == RelKind::Project) {
      bool identity = true;
      for (size_t i = 0; i < r->exprs.size(); ++i) identity &= r->exprs[i]->is_column() && r->exprs[i]->field_index == static_cast<int>(i);
      return identity && descend(r->inputs[0]);
    }
    return false;
  };
  Walk(root, [&](const RelPtr& r) { if (r->kind == RelKind::Filter) descend(r); });
  if (conj.empty()) return nullptr;
  if (conj.size() == 1) return conj[0];
  return Call("and", conj, DataType::Of(TypeId::Bool));
}

}  // namespace aster::planner
