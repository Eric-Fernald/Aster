#include "aster/integration/substrait_consumer.hpp"

#include <unordered_map>

#if ASTER_HAVE_SUBSTRAIT
#include <google/protobuf/util/json_util.h>
#include <substrait/plan.pb.h>
#endif

namespace aster::integration {

using namespace plan;

bool SubstraitConsumer::Available() {
#if ASTER_HAVE_SUBSTRAIT
  return true;
#else
  return false;
#endif
}

std::string SubstraitConsumer::NormalizeFunctionName(const std::string& name) {
  // "equal:i64_i64" -> "equal"; "add:opt_i64_i64" -> "add"
  size_t colon = name.find(':');
  std::string base = colon == std::string::npos ? name : name.substr(0, colon);
  static const std::unordered_map<std::string, std::string> alias = {
      {"eq", "equal"}, {"neq", "not_equal"}, {"ne", "not_equal"}, {"lte", "lte"}, {"gte", "gte"},
      {"less", "lt"}, {"greater", "gt"}, {"mul", "multiply"}, {"sub", "subtract"}, {"div", "divide"},
      {"mod", "modulus"}, {"count_star", "count_star"}, {"is_not_distinct_from", "equal"}};
  auto it = alias.find(base);
  return it == alias.end() ? base : it->second;
}

#if ASTER_HAVE_SUBSTRAIT
namespace {

struct Ctx {
  std::unordered_map<uint32_t, std::string> functions;
  TableResolver& resolver;
};

Result<DataType> ConvertType(const substrait::Type& t) {
  switch (t.kind_case()) {
    case substrait::Type::kBool: return DataType::Of(TypeId::Bool);
    case substrait::Type::kI8: return DataType::Of(TypeId::Int8);
    case substrait::Type::kI16: return DataType::Of(TypeId::Int16);
    case substrait::Type::kI32: return DataType::Of(TypeId::Int32);
    case substrait::Type::kI64: return DataType::Of(TypeId::Int64);
    case substrait::Type::kFp32: return DataType::Of(TypeId::Float32);
    case substrait::Type::kFp64: return DataType::Of(TypeId::Float64);
    case substrait::Type::kString: case substrait::Type::kVarchar: case substrait::Type::kFixedChar: return DataType::Of(TypeId::String);
    case substrait::Type::kBinary: return DataType::Of(TypeId::Binary);
    case substrait::Type::kDate: return DataType::Of(TypeId::Date32);
    case substrait::Type::kTimestamp: case substrait::Type::kTimestampTz: return DataType::Of(TypeId::Timestamp);
    case substrait::Type::kDecimal: return DataType::Decimal(t.decimal().precision(), t.decimal().scale());
    default: return Status::NotSupported("substrait type " + std::to_string(t.kind_case()));
  }
}

Result<ExprPtr> ConvertExpr(const substrait::Expression& e, Ctx& ctx, const Schema& input);

Result<ExprPtr> ConvertLiteral(const substrait::Expression::Literal& l) {
  switch (l.literal_type_case()) {
    case substrait::Expression::Literal::kBoolean: return Lit(l.boolean());
    case substrait::Expression::Literal::kI8: return Lit(int64_t(l.i8()));
    case substrait::Expression::Literal::kI16: return Lit(int64_t(l.i16()));
    case substrait::Expression::Literal::kI32: return Lit(int64_t(l.i32()));
    case substrait::Expression::Literal::kI64: return Lit(int64_t(l.i64()));
    case substrait::Expression::Literal::kFp32: return Lit(double(l.fp32()));
    case substrait::Expression::Literal::kFp64: return Lit(l.fp64());
    case substrait::Expression::Literal::kString: return Lit(l.string());
    case substrait::Expression::Literal::kVarChar: return Lit(l.var_char().value());
    case substrait::Expression::Literal::kFixedChar: return Lit(l.fixed_char());
    case substrait::Expression::Literal::kDate: { auto e = Lit(int64_t(l.date())); e->type = DataType::Of(TypeId::Date32); return e; }
    case substrait::Expression::Literal::kTimestamp: { auto e = Lit(int64_t(l.timestamp())); e->type = DataType::Of(TypeId::Timestamp); return e; }
    case substrait::Expression::Literal::kDecimal: {
      const auto& d = l.decimal();
      int64_t v = 0;
      std::memcpy(&v, d.value().data(), std::min<size_t>(8, d.value().size()));
      auto e = Lit(v);
      e->type = DataType::Decimal(d.precision(), d.scale());
      return e;
    }
    case substrait::Expression::Literal::kNull: {
      ASTER_ASSIGN_OR_RETURN(DataType t, ConvertType(l.null()));
      return LitNull(t);
    }
    default: return Status::NotSupported("substrait literal " + std::to_string(l.literal_type_case()));
  }
}

Result<ExprPtr> ConvertExpr(const substrait::Expression& e, Ctx& ctx, const Schema& input) {
  switch (e.rex_type_case()) {
    case substrait::Expression::kLiteral: return ConvertLiteral(e.literal());
    case substrait::Expression::kSelection: {
      const auto& s = e.selection();
      if (!s.has_direct_reference() || !s.direct_reference().has_struct_field()) return Status::NotSupported("non direct field reference");
      int idx = s.direct_reference().struct_field().field();
      DataType t = idx < static_cast<int>(input.fields.size()) ? input.fields[idx].type : DataType{};
      return Col(idx, t);
    }
    case substrait::Expression::kScalarFunction: {
      const auto& f = e.scalar_function();
      auto it = ctx.functions.find(f.function_reference());
      if (it == ctx.functions.end()) return Status::Invalid("unknown function anchor " + std::to_string(f.function_reference()));
      std::vector<ExprPtr> args;
      for (const auto& a : f.arguments()) {
        if (!a.has_value()) return Status::NotSupported("non value function argument");
        ASTER_ASSIGN_OR_RETURN(ExprPtr ae, ConvertExpr(a.value(), ctx, input));
        args.push_back(ae);
      }
      DataType out;
      if (f.has_output_type()) { auto t = ConvertType(f.output_type()); if (t.ok()) out = t.value(); }
      return Call(SubstraitConsumer::NormalizeFunctionName(it->second), std::move(args), out);
    }
    case substrait::Expression::kCast: {
      ASTER_ASSIGN_OR_RETURN(ExprPtr in, ConvertExpr(e.cast().input(), ctx, input));
      ASTER_ASSIGN_OR_RETURN(DataType t, ConvertType(e.cast().type()));
      return Cast(in, t);
    }
    case substrait::Expression::kIfThen: {
      const auto& it = e.if_then();
      std::vector<ExprPtr> args;
      for (const auto& c : it.ifs()) {
        ASTER_ASSIGN_OR_RETURN(ExprPtr cond, ConvertExpr(c.if_(), ctx, input));
        ASTER_ASSIGN_OR_RETURN(ExprPtr then, ConvertExpr(c.then(), ctx, input));
        args.push_back(cond); args.push_back(then);
      }
      ASTER_ASSIGN_OR_RETURN(ExprPtr els, ConvertExpr(it.else_(), ctx, input));
      args.push_back(els);
      return Call("if_then", std::move(args));
    }
    case substrait::Expression::kSingularOrList: {
      const auto& sl = e.singular_or_list();
      std::vector<ExprPtr> args;
      ASTER_ASSIGN_OR_RETURN(ExprPtr v, ConvertExpr(sl.value(), ctx, input));
      args.push_back(v);
      for (const auto& o : sl.options()) { ASTER_ASSIGN_OR_RETURN(ExprPtr oe, ConvertExpr(o, ctx, input)); args.push_back(oe); }
      return Call("in", std::move(args), DataType::Of(TypeId::Bool));
    }
    default: return Status::NotSupported("substrait expression " + std::to_string(e.rex_type_case()));
  }
}

Result<RelPtr> ConvertRel(const substrait::Rel& r, Ctx& ctx);

Result<RelPtr> ApplyEmit(RelPtr rel, const substrait::RelCommon& common) {
  if (!common.has_emit()) return rel;
  std::vector<ExprPtr> exprs;
  std::vector<std::string> names;
  for (int idx : common.emit().output_mapping()) {
    exprs.push_back(Col(idx, rel->output.fields.at(idx).type));
    names.push_back(rel->output.fields[idx].name);
  }
  return PlanBuilder::Project(rel, std::move(exprs), std::move(names));
}

Result<RelPtr> ConvertRel(const substrait::Rel& r, Ctx& ctx) {
  switch (r.rel_type_case()) {
    case substrait::Rel::kRead: {
      const auto& rd = r.read();
      if (!rd.has_named_table()) return Status::NotSupported("only named_table reads");
      std::vector<std::string> names(rd.named_table().names().begin(), rd.named_table().names().end());
      ASTER_ASSIGN_OR_RETURN(Schema schema, ctx.resolver.ResolveTable(names));
      std::vector<int> proj;
      if (rd.has_projection() && rd.projection().has_select())
        for (const auto& item : rd.projection().select().struct_items()) proj.push_back(item.field());
      RelPtr rel = PlanBuilder::Read(names.back(), schema, proj);
      if (rd.has_filter()) { ASTER_ASSIGN_OR_RETURN(rel->pushed_filter, ConvertExpr(rd.filter(), ctx, schema)); }
      return ApplyEmit(rel, rd.common());
    }
    case substrait::Rel::kFilter: {
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(r.filter().input(), ctx));
      ASTER_ASSIGN_OR_RETURN(ExprPtr pred, ConvertExpr(r.filter().condition(), ctx, in->output));
      return ApplyEmit(PlanBuilder::Filter(in, pred), r.filter().common());
    }
    case substrait::Rel::kProject: {
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(r.project().input(), ctx));
      // Substrait project output = input fields followed by new expressions.
      std::vector<ExprPtr> exprs;
      std::vector<std::string> names;
      for (size_t i = 0; i < in->output.fields.size(); ++i) { exprs.push_back(Col(int(i), in->output.fields[i].type)); names.push_back(in->output.fields[i].name); }
      for (const auto& e : r.project().expressions()) { ASTER_ASSIGN_OR_RETURN(ExprPtr ce, ConvertExpr(e, ctx, in->output)); exprs.push_back(ce); names.push_back("expr" + std::to_string(names.size())); }
      return ApplyEmit(PlanBuilder::Project(in, std::move(exprs), std::move(names)), r.project().common());
    }
    case substrait::Rel::kJoin: {
      const auto& j = r.join();
      ASTER_ASSIGN_OR_RETURN(RelPtr l, ConvertRel(j.left(), ctx));
      ASTER_ASSIGN_OR_RETURN(RelPtr rr, ConvertRel(j.right(), ctx));
      JoinType t = JoinType::Inner;
      switch (j.type()) {
        case substrait::JoinRel::JOIN_TYPE_LEFT: t = JoinType::Left; break;
        case substrait::JoinRel::JOIN_TYPE_RIGHT: t = JoinType::Right; break;
        case substrait::JoinRel::JOIN_TYPE_OUTER: t = JoinType::Full; break;
        case substrait::JoinRel::JOIN_TYPE_LEFT_SEMI: t = JoinType::Semi; break;
        case substrait::JoinRel::JOIN_TYPE_LEFT_ANTI: t = JoinType::Anti; break;
        default: break;
      }
      Schema combined = l->output;
      for (const auto& f : rr->output.fields) combined.fields.push_back(f);
      ASTER_ASSIGN_OR_RETURN(ExprPtr cond, ConvertExpr(j.expression(), ctx, combined));
      // Extract equi keys from a conjunction of equal(col, col).
      std::vector<int> lk, rk;
      std::vector<ExprPtr> stack{cond};
      int nl = static_cast<int>(l->output.fields.size());
      while (!stack.empty()) {
        ExprPtr e = stack.back(); stack.pop_back();
        if (e->is_call("and")) { for (auto& a : e->args) stack.push_back(a); continue; }
        if (e->is_call("equal") && e->args.size() == 2 && e->args[0]->is_column() && e->args[1]->is_column()) {
          int a = e->args[0]->field_index, b = e->args[1]->field_index;
          if (a < nl && b >= nl) { lk.push_back(a); rk.push_back(b - nl); }
          else if (b < nl && a >= nl) { lk.push_back(b); rk.push_back(a - nl); }
        }
      }
      RelPtr rel = PlanBuilder::Join(l, rr, t, lk, rk);
      rel->condition = cond;
      return ApplyEmit(rel, j.common());
    }
    case substrait::Rel::kAggregate: {
      const auto& a = r.aggregate();
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(a.input(), ctx));
      std::vector<ExprPtr> keys;
      if (a.groupings_size() > 1) return Status::NotSupported("grouping sets");
      if (a.groupings_size() == 1)
        for (const auto& g : a.groupings(0).grouping_expressions()) { ASTER_ASSIGN_OR_RETURN(ExprPtr k, ConvertExpr(g, ctx, in->output)); keys.push_back(k); }
      std::vector<AggregateFn> aggs;
      for (const auto& m : a.measures()) {
        const auto& f = m.measure();
        auto it = ctx.functions.find(f.function_reference());
        if (it == ctx.functions.end()) return Status::Invalid("unknown aggregate anchor");
        AggregateFn fn;
        fn.function = SubstraitConsumer::NormalizeFunctionName(it->second);
        fn.distinct = f.invocation() == substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT;
        if (fn.function == "count" && fn.distinct) fn.function = "count_distinct";
        if (fn.function == "count" && f.arguments_size() == 0) fn.function = "count_star";
        for (const auto& arg : f.arguments()) { ASTER_ASSIGN_OR_RETURN(ExprPtr ae, ConvertExpr(arg.value(), ctx, in->output)); fn.args.push_back(ae); }
        if (f.has_output_type()) { auto t = ConvertType(f.output_type()); if (t.ok()) fn.out_type = t.value(); }
        aggs.push_back(std::move(fn));
      }
      return ApplyEmit(PlanBuilder::Aggregate(in, std::move(keys), std::move(aggs)), a.common());
    }
    case substrait::Rel::kSort: {
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(r.sort().input(), ctx));
      std::vector<SortKey> keys;
      for (const auto& s : r.sort().sorts()) {
        SortKey k;
        ASTER_ASSIGN_OR_RETURN(k.expr, ConvertExpr(s.expr(), ctx, in->output));
        auto d = s.direction();
        k.ascending = d == substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST || d == substrait::SortField::SORT_DIRECTION_ASC_NULLS_LAST;
        k.nulls_first = d == substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST || d == substrait::SortField::SORT_DIRECTION_DESC_NULLS_FIRST;
        keys.push_back(k);
      }
      return ApplyEmit(PlanBuilder::Sort(in, std::move(keys)), r.sort().common());
    }
    case substrait::Rel::kFetch: {
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(r.fetch().input(), ctx));
      return ApplyEmit(PlanBuilder::Limit(in, r.fetch().offset(), r.fetch().count()), r.fetch().common());
    }
    case substrait::Rel::kExchange: {
      ASTER_ASSIGN_OR_RETURN(RelPtr in, ConvertRel(r.exchange().input(), ctx));
      std::vector<int> keys;
      ExchangeKind kind = ExchangeKind::HashPartition;
      if (r.exchange().has_broadcast()) kind = ExchangeKind::Broadcast;
      if (r.exchange().has_scatter_by_fields())
        for (const auto& f : r.exchange().scatter_by_fields().fields()) keys.push_back(f.direct_reference().struct_field().field());
      return PlanBuilder::Exchange(in, kind, keys);
    }
    default: return Status::NotSupported("substrait rel " + std::to_string(r.rel_type_case()));
  }
}

Result<RelPtr> Convert(const substrait::Plan& plan, TableResolver& resolver) {
  Ctx ctx{{}, resolver};
  for (const auto& ext : plan.extensions())
    if (ext.has_extension_function()) ctx.functions[ext.extension_function().function_anchor()] = ext.extension_function().name();
  if (plan.relations_size() == 0) return Status::Invalid("plan has no relations");
  const auto& root = plan.relations(0);
  RelPtr rel;
  if (root.has_root()) {
    ASTER_ASSIGN_OR_RETURN(rel, ConvertRel(root.root().input(), ctx));
    const auto& names = root.root().names();
    for (int i = 0; i < names.size() && i < static_cast<int>(rel->output.fields.size()); ++i) rel->output.fields[i].name = names[i];
  } else {
    ASTER_ASSIGN_OR_RETURN(rel, ConvertRel(root.rel(), ctx));
  }
  AssignNodeIds(rel);
  return rel;
}
}  // namespace

Result<RelPtr> SubstraitConsumer::FromBinary(const std::string& bytes, TableResolver& resolver) {
  substrait::Plan plan;
  if (!plan.ParseFromString(bytes)) return Status::Invalid("cannot parse substrait plan");
  return Convert(plan, resolver);
}

Result<RelPtr> SubstraitConsumer::FromJson(const std::string& json, TableResolver& resolver) {
  substrait::Plan plan;
  auto st = google::protobuf::util::JsonStringToMessage(json, &plan);
  if (!st.ok()) return Status::Invalid("cannot parse substrait json: " + st.ToString());
  return Convert(plan, resolver);
}
#else
Result<RelPtr> SubstraitConsumer::FromBinary(const std::string&, TableResolver&) {
  return Status::NotSupported("built without Substrait protobuf support");
}
Result<RelPtr> SubstraitConsumer::FromJson(const std::string&, TableResolver&) {
  return Status::NotSupported("built without Substrait protobuf support");
}
#endif

}  // namespace aster::integration
