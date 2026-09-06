#include "aster/integration/capability_registry.hpp"

#include <fstream>
#include <sstream>

namespace aster::integration {

const char* SupportName(Support s) {
  switch (s) {
    case Support::Gpu: return "gpu";
    case Support::Cpu: return "cpu";
    default: return "unsupported";
  }
}

namespace {
Support ParseSupport(const std::string& s) {
  if (s == "gpu") return Support::Gpu;
  if (s == "cpu") return Support::Cpu;
  return Support::Unsupported;
}
Support Min(Support a, Support b) { return static_cast<int>(a) > static_cast<int>(b) ? a : b; }
}  // namespace

CapabilityRegistry CapabilityRegistry::Defaults() { return CapabilityRegistry(); }

CapabilityRegistry::CapabilityRegistry() {
  CapabilityRegistry& r = *this;
  auto gpu = [&](const char* k, const char* note = "") { r.Set(k, Support::Gpu, note); };
  auto cpu = [&](const char* k, const char* note = "") { r.Set(k, Support::Cpu, note); };
  auto no = [&](const char* k, const char* note = "") { r.Set(k, Support::Unsupported, note); };

  for (const char* k : {"rel:read", "rel:filter", "rel:project", "rel:join", "rel:aggregate", "rel:sort", "rel:limit", "rel:exchange", "rel:externalinput"}) gpu(k);
  cpu("rel:window", "phase 2");
  for (const char* k : {"join:inner", "join:left", "join:semi", "join:anti"}) gpu(k);
  cpu("join:right", "rewrite as left join upstream");
  cpu("join:full");
  for (const char* k : {"type:bool", "type:i8", "type:i16", "type:i32", "type:i64", "type:u8", "type:u16", "type:u32", "type:u64",
                        "type:f32", "type:f64", "type:decimal64", "type:date32", "type:timestamp", "type:string", "type:vector"}) gpu(k);
  cpu("type:binary");
  no("type:null");
  for (const char* k : {"equal", "not_equal", "lt", "lte", "gt", "gte", "and", "or", "not", "add", "subtract", "multiply", "divide",
                        "modulus", "negate", "is_null", "is_not_null", "between", "in", "coalesce", "if_then", "abs", "like",
                        "starts_with", "ends_with", "contains", "substring", "length", "lower", "upper", "concat", "extract", "year",
                        "cast", "round", "floor", "ceil", "hash", "vector_distance"})
    r.Set(std::string("fn:") + k, Support::Gpu);
  for (const char* k : {"regexp_match", "regexp_replace", "trim", "ltrim", "rtrim", "date_trunc", "strftime", "strptime", "split_part", "levenshtein"})
    r.Set(std::string("fn:") + k, Support::Cpu, "string kernel not yet ported");
  for (const char* k : {"sum", "count", "count_star", "min", "max", "avg", "count_distinct"}) r.Set(std::string("agg:") + k, Support::Gpu);
  for (const char* k : {"median", "quantile", "string_agg", "approx_count_distinct", "stddev", "variance"}) r.Set(std::string("agg:") + k, Support::Cpu);
  for (const char* k : {"row_number", "rank", "dense_rank", "sum", "count", "min", "max", "avg", "lag", "lead"}) r.Set(std::string("win:") + k, Support::Cpu, "phase 2");
}

void CapabilityRegistry::Set(const std::string& key, Support s, std::string note) {
  table_[key] = CapabilityEntry{key, s, std::move(note)};
}

Support CapabilityRegistry::Get(const std::string& key) const {
  auto it = table_.find(key);
  return it == table_.end() ? Support::Unsupported : it->second.support;
}

Support CapabilityRegistry::Relation(plan::RelKind k) const {
  std::string name = plan::RelKindName(k);
  for (auto& c : name) c = static_cast<char>(tolower(c));
  return Get("rel:" + name);
}

Support CapabilityRegistry::Type(TypeId t) const { return Get(std::string("type:") + TypeName(t)); }

Support CapabilityRegistry::JoinKind(plan::JoinType t) const { return Get(std::string("join:") + plan::JoinTypeName(t)); }

Support CapabilityRegistry::CheckExpr(const plan::Expr& e, std::string* reason) const {
  Support s = Support::Gpu;
  if (e.kind == plan::ExprKind::Call) {
    Support f = Function(e.function);
    if (f != Support::Gpu && reason) *reason = "fn:" + e.function + " is " + SupportName(f);
    s = Min(s, f);
  }
  if (e.type.id != TypeId::Null) {
    Support t = Type(e.type.id);
    if (t != Support::Gpu && reason && reason->empty()) *reason = std::string("type:") + TypeName(e.type.id) + " is " + SupportName(t);
    s = Min(s, t);
  }
  for (const auto& a : e.args) {
    std::string r;
    Support as = CheckExpr(*a, &r);
    if (as != Support::Gpu && reason && reason->empty()) *reason = r;
    s = Min(s, as);
  }
  return s;
}

Support CapabilityRegistry::CheckNode(const plan::Rel& rel, std::string* reason) const {
  std::string why;
  Support s = Relation(rel.kind);
  if (s != Support::Gpu) why = std::string("rel:") + plan::RelKindName(rel.kind) + " is " + SupportName(s);
  auto fold = [&](Support x, const std::string& r) {
    if (x != Support::Gpu && why.empty()) why = r;
    s = Min(s, x);
  };
  for (const auto& f : rel.output.fields) fold(Type(f.type.id), std::string("type:") + TypeName(f.type.id));
  auto check_expr = [&](const plan::ExprPtr& e) {
    if (!e) return;
    std::string r;
    Support x = CheckExpr(*e, &r);
    fold(x, r);
  };
  check_expr(rel.predicate);
  check_expr(rel.pushed_filter);
  check_expr(rel.condition);
  for (const auto& e : rel.exprs) check_expr(e);
  for (const auto& e : rel.group_keys) check_expr(e);
  for (const auto& k : rel.sort_keys) check_expr(k.expr);
  if (rel.kind == plan::RelKind::Join) fold(JoinKind(rel.join_type), std::string("join:") + plan::JoinTypeName(rel.join_type));
  for (const auto& a : rel.aggregates) {
    fold(Aggregate(a.function), "agg:" + a.function);
    for (const auto& e : a.args) check_expr(e);
  }
  for (const auto& w : rel.windows) {
    fold(Get("win:" + w.function), "win:" + w.function);
    for (const auto& e : w.args) check_expr(e);
  }
  if (reason) *reason = why;
  return s;
}

Status CapabilityRegistry::LoadTsv(const std::string& path) {
  std::ifstream f(path);
  if (!f) return Status::NotFound(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string key, sup, note;
    std::getline(is, key, '\t');
    std::getline(is, sup, '\t');
    std::getline(is, note);
    if (key.empty()) continue;
    Set(key, ParseSupport(sup), note);
  }
  return Status::OK();
}

Status CapabilityRegistry::SaveTsv(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return Status::IoError("cannot write " + path);
  f << "# key\tsupport\tnote\n";
  for (const auto& [k, e] : table_) f << k << "\t" << SupportName(e.support) << "\t" << e.note << "\n";
  return Status::OK();
}

std::vector<CapabilityEntry> CapabilityRegistry::Entries() const {
  std::vector<CapabilityEntry> out;
  for (const auto& [k, e] : table_) out.push_back(e);
  return out;
}

std::vector<CapabilityEntry> CapabilityRegistry::Gaps() const {
  std::vector<CapabilityEntry> out;
  for (const auto& [k, e] : table_)
    if (e.support != Support::Gpu) out.push_back(e);
  return out;
}

}  // namespace aster::integration
