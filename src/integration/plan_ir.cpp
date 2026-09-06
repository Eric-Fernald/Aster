#include "aster/integration/plan_ir.hpp"

#include <functional>
#include <sstream>

namespace aster::plan {

ExprPtr Lit(int64_t v) { auto e = std::make_shared<Expr>(); e->type = DataType::Of(TypeId::Int64); e->literal = v; return e; }
ExprPtr Lit(double v) { auto e = std::make_shared<Expr>(); e->type = DataType::Of(TypeId::Float64); e->literal = v; return e; }
ExprPtr Lit(std::string v) { auto e = std::make_shared<Expr>(); e->type = DataType::Of(TypeId::String); e->literal = std::move(v); return e; }
ExprPtr Lit(bool v) { auto e = std::make_shared<Expr>(); e->type = DataType::Of(TypeId::Bool); e->literal = v; return e; }
ExprPtr LitNull(DataType t) { auto e = std::make_shared<Expr>(); e->type = t; e->literal = NullLiteral{}; return e; }
ExprPtr Col(int index, DataType t) { auto e = std::make_shared<Expr>(); e->kind = ExprKind::ColumnRef; e->field_index = index; e->type = t; return e; }
ExprPtr Call(std::string fn, std::vector<ExprPtr> args, DataType out) {
  auto e = std::make_shared<Expr>();
  e->kind = ExprKind::Call; e->function = std::move(fn); e->args = std::move(args); e->type = out;
  return e;
}
ExprPtr Cast(ExprPtr in, DataType to) {
  auto e = std::make_shared<Expr>();
  e->kind = ExprKind::Cast; e->args = {std::move(in)}; e->type = to;
  return e;
}

std::string Expr::ToString() const {
  switch (kind) {
    case ExprKind::Literal:
      if (std::holds_alternative<NullLiteral>(literal)) return "NULL";
      if (auto b = std::get_if<bool>(&literal)) return *b ? "true" : "false";
      if (auto i = std::get_if<int64_t>(&literal)) return std::to_string(*i);
      if (auto d = std::get_if<double>(&literal)) return std::to_string(*d);
      return "'" + std::get<std::string>(literal) + "'";
    case ExprKind::ColumnRef: return "$" + std::to_string(field_index);
    case ExprKind::Cast: return "cast(" + args[0]->ToString() + " as " + TypeToString(type) + ")";
    case ExprKind::Call: {
      std::string s = function + "(";
      for (size_t i = 0; i < args.size(); ++i) { if (i) s += ", "; s += args[i]->ToString(); }
      return s + ")" + (encoded_domain ? "[enc]" : "");
    }
  }
  return "?";
}

const char* RelKindName(RelKind k) {
  static const char* names[] = {"Read", "Filter", "Project", "Join", "Aggregate", "Sort", "Limit", "Window", "Exchange", "ExternalInput"};
  return names[static_cast<int>(k)];
}

const char* JoinTypeName(JoinType t) {
  static const char* names[] = {"inner", "left", "right", "full", "semi", "anti"};
  return names[static_cast<int>(t)];
}

std::string Rel::ToString(int indent) const {
  std::ostringstream os;
  std::string pad(indent * 2, ' ');
  os << pad << "#" << node_id << " " << RelKindName(kind);
  if (placement == Placement::Cpu) os << " [cpu: " << fallback_reason << "]";
  else if (placement == Placement::Gpu) os << " [gpu]";
  switch (kind) {
    case RelKind::Read: os << " table=" << table; break;
    case RelKind::Filter: os << " " << predicate->ToString(); break;
    case RelKind::Project: for (const auto& e : exprs) os << " " << e->ToString(); break;
    case RelKind::Join: os << " " << JoinTypeName(join_type) << " keys=" << left_keys.size(); break;
    case RelKind::Aggregate: os << " keys=" << group_keys.size() << " aggs=" << aggregates.size(); break;
    case RelKind::Sort: os << " keys=" << sort_keys.size(); break;
    case RelKind::Limit: os << " offset=" << offset << " count=" << count; break;
    case RelKind::Window: os << " fns=" << windows.size(); break;
    case RelKind::Exchange: os << " kind=" << int(exchange_kind); break;
    case RelKind::ExternalInput: os << " batches=" << external_batches.size(); break;
  }
  os << " -> [" << output.ToString() << "]\n";
  for (const auto& in : inputs) os << in->ToString(indent + 1);
  return os.str();
}

void Walk(const RelPtr& root, const std::function<void(const RelPtr&)>& fn) {
  if (!root) return;
  for (const auto& in : root->inputs) Walk(in, fn);
  fn(root);
}

void AssignNodeIds(const RelPtr& root) {
  int id = 0;
  Walk(root, [&](const RelPtr& r) { r->node_id = id++; });
}

std::vector<RelPtr> Collect(const RelPtr& root, RelKind kind) {
  std::vector<RelPtr> out;
  Walk(root, [&](const RelPtr& r) { if (r->kind == kind) out.push_back(r); });
  return out;
}

DataType AggregateOutputType(const std::string& fn, DataType in) {
  if (fn == "count" || fn == "count_distinct" || fn == "count_star") return DataType::Of(TypeId::Int64);
  if (fn == "avg") return DataType::Of(TypeId::Float64);
  if (fn == "sum") {
    if (in.id == TypeId::Float32 || in.id == TypeId::Float64) return DataType::Of(TypeId::Float64);
    if (in.id == TypeId::Decimal64) return in;
    return DataType::Of(TypeId::Int64);
  }
  return in;
}

DataType InferType(const Expr& e, const Schema& input) {
  switch (e.kind) {
    case ExprKind::Literal: return e.type;
    case ExprKind::Cast: return e.type;
    case ExprKind::ColumnRef:
      if (e.field_index >= 0 && e.field_index < static_cast<int>(input.fields.size())) return input.fields[e.field_index].type;
      return e.type;
    case ExprKind::Call: {
      if (e.type.id != TypeId::Null) return e.type;
      static const char* bools[] = {"equal", "not_equal", "lt", "lte", "gt", "gte", "and", "or", "not", "like", "is_null", "is_not_null", "in", "between", "starts_with", "ends_with", "contains"};
      for (const char* b : bools) if (e.function == b) return DataType::Of(TypeId::Bool);
      if (e.function == "concat" || e.function == "substring" || e.function == "lower" || e.function == "upper" || e.function == "trim")
        return DataType::Of(TypeId::String);
      if (e.function == "length" || e.function == "extract" || e.function == "year") return DataType::Of(TypeId::Int64);
      DataType best;
      for (const auto& a : e.args) {
        DataType t = InferType(*a, input);
        if (best.id == TypeId::Null || t.id == TypeId::Float64 || (t.id == TypeId::Float32 && best.id != TypeId::Float64) ||
            (t.id == TypeId::Decimal64 && !(best.id == TypeId::Float64 || best.id == TypeId::Float32)) ||
            (t.id == TypeId::Int64 && IsIntegral(best.id)))
          best = t;
      }
      if (e.function == "divide" && IsIntegral(best.id)) return DataType::Of(TypeId::Float64);
      return best;
    }
  }
  return e.type;
}

namespace {
RelPtr MakeRel(RelKind k, std::vector<RelPtr> inputs) {
  auto r = std::make_shared<Rel>();
  r->kind = k;
  r->inputs = std::move(inputs);
  return r;
}
std::string NameOr(const std::vector<std::string>& names, size_t i, const std::string& fallback) {
  return i < names.size() ? names[i] : fallback;
}
}  // namespace

RelPtr PlanBuilder::Read(std::string table, Schema schema, std::vector<int> projection) {
  auto r = MakeRel(RelKind::Read, {});
  r->table = std::move(table);
  if (projection.empty()) {
    r->output = schema;
    for (size_t i = 0; i < schema.fields.size(); ++i) r->projection.push_back(static_cast<int>(i));
  } else {
    for (int i : projection) r->output.fields.push_back(schema.fields.at(i));
    r->projection = std::move(projection);
  }
  return r;
}

RelPtr PlanBuilder::External(Schema schema, std::vector<RecordBatchPtr> batches) {
  auto r = MakeRel(RelKind::ExternalInput, {});
  r->output = std::move(schema);
  r->external_batches = std::move(batches);
  r->placement = Placement::Cpu;
  return r;
}

RelPtr PlanBuilder::Filter(RelPtr in, ExprPtr predicate) {
  auto r = MakeRel(RelKind::Filter, {in});
  r->output = in->output;
  r->predicate = std::move(predicate);
  return r;
}

RelPtr PlanBuilder::Project(RelPtr in, std::vector<ExprPtr> exprs, std::vector<std::string> names) {
  auto r = MakeRel(RelKind::Project, {in});
  for (size_t i = 0; i < exprs.size(); ++i) {
    DataType t = InferType(*exprs[i], in->output);
    std::string fallback = exprs[i]->is_column() ? in->output.fields[exprs[i]->field_index].name : "expr" + std::to_string(i);
    r->output.fields.push_back({NameOr(names, i, fallback), t, true});
  }
  r->exprs = std::move(exprs);
  return r;
}

RelPtr PlanBuilder::Join(RelPtr l, RelPtr rr, JoinType t, std::vector<int> lkeys, std::vector<int> rkeys) {
  auto r = MakeRel(RelKind::Join, {l, rr});
  r->join_type = t;
  r->left_keys = std::move(lkeys);
  r->right_keys = std::move(rkeys);
  r->output = l->output;
  if (t != JoinType::Semi && t != JoinType::Anti)
    for (const auto& f : rr->output.fields) r->output.fields.push_back(f);
  return r;
}

RelPtr PlanBuilder::Aggregate(RelPtr in, std::vector<ExprPtr> keys, std::vector<AggregateFn> aggs, std::vector<std::string> names) {
  auto r = MakeRel(RelKind::Aggregate, {in});
  size_t n = 0;
  for (const auto& k : keys) {
    std::string fallback = k->is_column() ? in->output.fields[k->field_index].name : "key" + std::to_string(n);
    r->output.fields.push_back({NameOr(names, n++, fallback), InferType(*k, in->output), true});
  }
  for (auto& a : aggs) {
    DataType in_t = a.args.empty() ? DataType::Of(TypeId::Int64) : InferType(*a.args[0], in->output);
    if (a.out_type.id == TypeId::Null) a.out_type = AggregateOutputType(a.function, in_t);
    r->output.fields.push_back({NameOr(names, n++, a.function), a.out_type, true});
  }
  r->group_keys = std::move(keys);
  r->aggregates = std::move(aggs);
  return r;
}

RelPtr PlanBuilder::Sort(RelPtr in, std::vector<SortKey> keys) {
  auto r = MakeRel(RelKind::Sort, {in});
  r->output = in->output;
  r->sort_keys = std::move(keys);
  return r;
}

RelPtr PlanBuilder::Limit(RelPtr in, int64_t offset, int64_t count) {
  auto r = MakeRel(RelKind::Limit, {in});
  r->output = in->output;
  r->offset = offset;
  r->count = count;
  return r;
}

RelPtr PlanBuilder::Window(RelPtr in, std::vector<WindowFn> fns, std::vector<std::string> names) {
  auto r = MakeRel(RelKind::Window, {in});
  r->output = in->output;
  for (size_t i = 0; i < fns.size(); ++i) {
    if (fns[i].out_type.id == TypeId::Null) {
      DataType in_t = fns[i].args.empty() ? DataType::Of(TypeId::Int64) : InferType(*fns[i].args[0], in->output);
      fns[i].out_type = (fns[i].function == "row_number" || fns[i].function == "rank" || fns[i].function == "dense_rank")
                            ? DataType::Of(TypeId::Int64) : AggregateOutputType(fns[i].function, in_t);
    }
    r->output.fields.push_back({NameOr(names, i, fns[i].function), fns[i].out_type, true});
  }
  r->windows = std::move(fns);
  return r;
}

RelPtr PlanBuilder::Exchange(RelPtr in, ExchangeKind kind, std::vector<int> keys) {
  auto r = MakeRel(RelKind::Exchange, {in});
  r->output = in->output;
  r->exchange_kind = kind;
  r->exchange_keys = std::move(keys);
  return r;
}

}  // namespace aster::plan
