#include "aster/planner/predicate_rewrite.hpp"

namespace aster::planner {

using namespace plan;

namespace {

ExprPtr Bool(bool v) { return Lit(v); }

bool ColLit(const Expr& e, int* col, const Expr** lit, bool* flipped) {
  if (e.args.size() != 2) return false;
  if (e.args[0]->is_column() && e.args[1]->is_literal()) { *col = e.args[0]->field_index; *lit = e.args[1].get(); *flipped = false; return true; }
  if (e.args[1]->is_column() && e.args[0]->is_literal()) { *col = e.args[1]->field_index; *lit = e.args[0].get(); *flipped = true; return true; }
  return false;
}

std::string FlipOp(const std::string& op) {
  if (op == "lt") return "gt";
  if (op == "lte") return "gte";
  if (op == "gt") return "lt";
  if (op == "gte") return "lte";
  return op;
}

ExprPtr EncodedCol(int idx, uint64_t dict_id) {
  ExprPtr c = Col(idx, DataType::Of(TypeId::Int32));
  c->encoded_domain = true;
  c->dictionary_id = dict_id;
  return c;
}

ExprPtr RewriteDictionary(const Expr& e, int col, const Expr& lit, const std::string& op, const storage::DictionaryView& d) {
  auto s = std::get_if<std::string>(&lit.literal);
  if (!s) return nullptr;
  if (op == "equal" || op == "not_equal") {
    int32_t code = d.LookupCode(*s);
    if (code < 0) return Bool(op == "not_equal");
    ExprPtr out = Call(op, {EncodedCol(col, 0), Lit(int64_t(code))}, DataType::Of(TypeId::Bool));
    out->encoded_domain = true;
    return out;
  }
  if (op == "in") {
    std::vector<ExprPtr> args{EncodedCol(col, 0)};
    for (size_t i = 1; i < e.args.size(); ++i) {
      auto v = std::get_if<std::string>(&e.args[i]->literal);
      if (!v) return nullptr;
      int32_t code = d.LookupCode(*v);
      if (code >= 0) args.push_back(Lit(int64_t(code)));
    }
    if (args.size() == 1) return Bool(false);
    ExprPtr out = Call("in", std::move(args), DataType::Of(TypeId::Bool));
    out->encoded_domain = true;
    return out;
  }
  return nullptr;
}

ExprPtr RewriteFor(int col, const Expr& lit, const std::string& op, const storage::ForView& f) {
  int64_t v;
  if (auto i = std::get_if<int64_t>(&lit.literal)) v = *i;
  else if (auto d = std::get_if<double>(&lit.literal)) v = static_cast<int64_t>(*d);
  else return nullptr;
  int64_t max_v = f.reference + (f.bit_width >= 63 ? INT64_MAX - f.reference : (int64_t(1) << f.bit_width) - 1);
  // Constant fold when the literal falls outside the representable range of this chunk.
  if (op == "equal") { if (v < f.reference || v > max_v) return Bool(false); }
  if (op == "not_equal") { if (v < f.reference || v > max_v) return Bool(true); }
  if (op == "lt") { if (v <= f.reference) return Bool(false); if (v > max_v) return Bool(true); }
  if (op == "lte") { if (v < f.reference) return Bool(false); if (v >= max_v) return Bool(true); }
  if (op == "gt") { if (v >= max_v) return Bool(false); if (v < f.reference) return Bool(true); }
  if (op == "gte") { if (v > max_v) return Bool(false); if (v <= f.reference) return Bool(true); }
  ExprPtr c = Col(col, DataType::Of(TypeId::UInt64));
  c->encoded_domain = true;
  ExprPtr out = Call(op, {c, Lit(v - f.reference)}, DataType::Of(TypeId::Bool));
  out->encoded_domain = true;
  return out;
}

}  // namespace

std::optional<bool> PredicateRewriter::ConstantValue(const Expr& e) {
  if (e.kind != ExprKind::Literal) return std::nullopt;
  if (auto b = std::get_if<bool>(&e.literal)) return *b;
  return std::nullopt;
}

bool PredicateRewriter::IsRewritable(const Expr& pred) {
  if (pred.kind != ExprKind::Call) return false;
  const std::string& f = pred.function;
  if (f == "and" || f == "or" || f == "not") {
    for (const auto& a : pred.args) if (IsRewritable(*a)) return true;
    return false;
  }
  int col; const Expr* lit; bool flipped;
  if (f == "in") return pred.args.size() >= 2 && pred.args[0]->is_column();
  return (f == "equal" || f == "not_equal" || f == "lt" || f == "lte" || f == "gt" || f == "gte") && ColLit(pred, &col, &lit, &flipped);
}

ExprPtr PredicateRewriter::Rewrite(const Expr& pred, const std::map<int, EncodedDomain>& domains) {
  if (pred.kind != ExprKind::Call) return std::make_shared<Expr>(pred);
  const std::string& f = pred.function;
  if (f == "and" || f == "or") {
    std::vector<ExprPtr> args;
    for (const auto& a : pred.args) {
      ExprPtr r = Rewrite(*a, domains);
      auto c = ConstantValue(*r);
      if (c) {
        if (f == "and" && !*c) return Bool(false);
        if (f == "or" && *c) return Bool(true);
        continue;  // neutral element
      }
      args.push_back(r);
    }
    if (args.empty()) return Bool(f == "and");
    if (args.size() == 1) return args[0];
    return Call(f, std::move(args), DataType::Of(TypeId::Bool));
  }
  if (f == "not" && pred.args.size() == 1) {
    ExprPtr r = Rewrite(*pred.args[0], domains);
    auto c = ConstantValue(*r);
    if (c) return Bool(!*c);
    return Call("not", {r}, DataType::Of(TypeId::Bool));
  }
  int col; const Expr* lit; bool flipped = false;
  std::string op = f;
  if (f == "in") {
    if (pred.args.size() < 2 || !pred.args[0]->is_column()) return std::make_shared<Expr>(pred);
    col = pred.args[0]->field_index;
    lit = pred.args[1].get();
  } else if (!ColLit(pred, &col, &lit, &flipped)) {
    return std::make_shared<Expr>(pred);
  }
  if (flipped) op = FlipOp(op);
  auto it = domains.find(col);
  if (it == domains.end()) return std::make_shared<Expr>(pred);
  const EncodedDomain& d = it->second;
  ExprPtr out;
  if (d.encoding == storage::Encoding::Dictionary && d.dictionary) out = RewriteDictionary(pred, col, *lit, op, *d.dictionary);
  else if (d.encoding == storage::Encoding::ForBitPack && d.frame && op != "in") out = RewriteFor(col, *lit, op, *d.frame);
  return out ? out : std::make_shared<Expr>(pred);
}

}  // namespace aster::planner
