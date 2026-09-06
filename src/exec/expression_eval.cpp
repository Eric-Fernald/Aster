#include "aster/exec/expression_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "aster/common/hash.hpp"

namespace aster::exec {

using namespace plan;

namespace {

inline bool IsFloat(TypeId t) { return t == TypeId::Float32 || t == TypeId::Float64; }
inline bool IsStr(TypeId t) { return t == TypeId::String || t == TypeId::Binary; }

double AsDouble(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::Int64: case TypeId::Timestamp: return double(c.Values<int64_t>()[i]);
    case TypeId::UInt64: return double(c.Values<uint64_t>()[i]);
    case TypeId::Float32: return c.Values<float>()[i];
    case TypeId::Float64: return c.Values<double>()[i];
    case TypeId::Decimal64: return double(c.Values<int64_t>()[i]) / std::pow(10.0, c.type.width);
    default: return 0;
  }
}

int64_t AsInt(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::Int64: case TypeId::Timestamp: case TypeId::Decimal64: return c.Values<int64_t>()[i];
    case TypeId::UInt64: return static_cast<int64_t>(c.Values<uint64_t>()[i]);
    case TypeId::Float32: return static_cast<int64_t>(c.Values<float>()[i]);
    case TypeId::Float64: return static_cast<int64_t>(c.Values<double>()[i]);
    default: return 0;
  }
}

Column Broadcast(const Expr& lit, int64_t n) {
  if (std::holds_alternative<NullLiteral>(lit.literal)) {
    Column c = Column::Empty(lit.type.id == TypeId::Null ? DataType::Of(TypeId::Int64) : lit.type);
    c.length = n;
    c.values = Buffer::AllocateHost(std::max<size_t>(1, TypeByteWidth(c.type)) * n);
    if (!IsFixedWidth(c.type.id)) c.offsets = Buffer::AllocateHost((n + 1) * sizeof(int32_t));
    std::vector<bool> valid(n, false);
    c.validity = MakeValidity(valid, &c.null_count);
    return c;
  }
  if (auto b = std::get_if<bool>(&lit.literal)) return MakeColumn<uint8_t>(TypeId::Bool, std::vector<uint8_t>(n, *b));
  if (auto i = std::get_if<int64_t>(&lit.literal)) {
    Column c = MakeColumn<int64_t>(TypeId::Int64, std::vector<int64_t>(n, *i));
    if (lit.type.id == TypeId::Date32) { c = MakeColumn<int32_t>(TypeId::Date32, std::vector<int32_t>(n, static_cast<int32_t>(*i))); }
    else if (lit.type.id == TypeId::Decimal64 || lit.type.id == TypeId::Timestamp) c.type = lit.type;
    return c;
  }
  if (auto d = std::get_if<double>(&lit.literal)) return MakeColumn<double>(TypeId::Float64, std::vector<double>(n, *d));
  return MakeStringColumn(std::vector<std::string>(n, std::get<std::string>(lit.literal)));
}

Column BoolColumn(int64_t n) {
  Column c = MakeColumn<uint8_t>(TypeId::Bool, std::vector<uint8_t>(n, 0));
  return c;
}

Column WithNulls(Column c, const std::vector<bool>& valid) {
  bool any = false;
  for (bool v : valid) if (!v) { any = true; break; }
  if (any) c.validity = MakeValidity(valid, &c.null_count);
  return c;
}

std::vector<bool> ValidMask(const std::vector<Column>& cols, int64_t n) {
  std::vector<bool> v(n, true);
  for (const auto& c : cols) {
    if (!c.validity) continue;
    for (int64_t i = 0; i < n; ++i) if (!c.IsValid(i)) v[i] = false;
  }
  return v;
}

int CompareStr(std::string_view a, std::string_view b) { return a.compare(b); }

int64_t DaysToYear(int32_t days) {
  // civil from days (Howard Hinnant)
  int64_t z = days + 719468;
  int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  int64_t doe = z - era * 146097;
  int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int64_t y = yoe + era * 400;
  int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  int64_t mp = (5 * doy + 2) / 153;
  int64_t m = mp < 10 ? mp + 3 : mp - 9;
  return m <= 2 ? y + 1 : y;
}

int64_t DaysToMonth(int32_t days) {
  int64_t z = days + 719468;
  int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  int64_t doe = z - era * 146097;
  int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  int64_t mp = (5 * doy + 2) / 153;
  return mp < 10 ? mp + 3 : mp - 9;
}

}  // namespace

bool ExpressionEvaluator::LikeMatch(std::string_view s, std::string_view p) {
  size_t si = 0, pi = 0, star_p = std::string::npos, star_s = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) { ++si; ++pi; }
    else if (pi < p.size() && p[pi] == '%') { star_p = pi++; star_s = si; }
    else if (star_p != std::string::npos) { pi = star_p + 1; si = ++star_s; }
    else return false;
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

Result<Column> ExpressionEvaluator::Cast(const Column& c, DataType to) {
  if (c.type == to) return c;
  int64_t n = c.length;
  std::vector<bool> valid(n);
  for (int64_t i = 0; i < n; ++i) valid[i] = c.IsValid(i);
  Column src = c.is_dictionary_encoded() ? c.DecodeDictionary() : c;
  if (to.id == TypeId::String) {
    std::vector<std::string> out(n);
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      if (IsStr(src.type.id)) out[i] = std::string(src.GetString(i));
      else if (IsFloat(src.type.id) || src.type.id == TypeId::Decimal64) out[i] = std::to_string(AsDouble(src, i));
      else out[i] = std::to_string(AsInt(src, i));
    }
    return WithNulls(MakeStringColumn(out), valid);
  }
  auto parse_num = [&](int64_t i, double* d) {
    if (IsStr(src.type.id)) { *d = std::strtod(std::string(src.GetString(i)).c_str(), nullptr); }
    else *d = AsDouble(src, i);
  };
  Column out;
  out.type = to;
  out.length = n;
  out.values = Buffer::AllocateHost(TypeByteWidth(to) * n);
  for (int64_t i = 0; i < n; ++i) {
    if (!valid[i]) continue;
    double d; parse_num(i, &d);
    switch (to.id) {
      case TypeId::Bool: case TypeId::UInt8: out.MutableValues<uint8_t>()[i] = static_cast<uint8_t>(d); break;
      case TypeId::Int8: out.MutableValues<int8_t>()[i] = static_cast<int8_t>(d); break;
      case TypeId::Int16: out.MutableValues<int16_t>()[i] = static_cast<int16_t>(d); break;
      case TypeId::UInt16: out.MutableValues<uint16_t>()[i] = static_cast<uint16_t>(d); break;
      case TypeId::Int32: case TypeId::Date32: out.MutableValues<int32_t>()[i] = static_cast<int32_t>(d); break;
      case TypeId::UInt32: out.MutableValues<uint32_t>()[i] = static_cast<uint32_t>(d); break;
      case TypeId::Int64: case TypeId::Timestamp: out.MutableValues<int64_t>()[i] = IsStr(src.type.id) ? static_cast<int64_t>(d) : (IsFloat(src.type.id) || src.type.id == TypeId::Decimal64 ? static_cast<int64_t>(d) : AsInt(src, i)); break;
      case TypeId::UInt64: out.MutableValues<uint64_t>()[i] = static_cast<uint64_t>(d); break;
      case TypeId::Float32: out.MutableValues<float>()[i] = static_cast<float>(d); break;
      case TypeId::Float64: out.MutableValues<double>()[i] = d; break;
      case TypeId::Decimal64: out.MutableValues<int64_t>()[i] = static_cast<int64_t>(std::llround(d * std::pow(10.0, to.width))); break;
      default: return Status::NotSupported("cast to " + TypeToString(to));
    }
  }
  return WithNulls(std::move(out), valid);
}

Result<Column> ExpressionEvaluator::Evaluate(const Expr& e, const RecordBatch& in) {
  const int64_t n = in.num_rows();
  switch (e.kind) {
    case ExprKind::Literal: return Broadcast(e, n);
    case ExprKind::ColumnRef:
      if (e.field_index < 0 || e.field_index >= static_cast<int>(in.columns.size()))
        return Status::Invalid("column ref out of range: " + std::to_string(e.field_index));
      return in.columns[e.field_index];
    case ExprKind::Cast: {
      ASTER_ASSIGN_OR_RETURN(Column c, Evaluate(*e.args[0], in));
      return Cast(c, e.type);
    }
    case ExprKind::Call: break;
  }
  const std::string& f = e.function;

  // Dictionary fast path: equality against a string literal compares codes, no decode.
  if ((f == "equal" || f == "not_equal") && e.args.size() == 2) {
    const Expr* col = nullptr; const Expr* lit = nullptr;
    if (e.args[0]->is_column() && e.args[1]->is_literal()) { col = e.args[0].get(); lit = e.args[1].get(); }
    if (e.args[1]->is_column() && e.args[0]->is_literal()) { col = e.args[1].get(); lit = e.args[0].get(); }
    if (col && lit && col->field_index < static_cast<int>(in.columns.size())) {
      const Column& c = in.columns[col->field_index];
      if (auto s = std::get_if<std::string>(&lit->literal); s && c.is_dictionary_encoded()) {
        int32_t code = -1;
        for (int64_t k = 0; k < c.dictionary_size(); ++k) if (c.DictionaryEntry(static_cast<int32_t>(k)) == *s) { code = static_cast<int32_t>(k); break; }
        Column out = BoolColumn(n);
        const int32_t* codes = c.Values<int32_t>();
        bool eq = f == "equal";
        for (int64_t i = 0; i < n; ++i) out.MutableValues<uint8_t>()[i] = (codes[i] == code) == eq;
        if (c.validity) { std::vector<bool> valid(n); for (int64_t i = 0; i < n; ++i) valid[i] = c.IsValid(i); out = WithNulls(std::move(out), valid); }
        return out;
      }
    }
  }

  std::vector<Column> args;
  for (const auto& a : e.args) { ASTER_ASSIGN_OR_RETURN(Column c, Evaluate(*a, in)); args.push_back(std::move(c)); }

  auto need = [&](size_t k) -> Status { return args.size() >= k ? Status::OK() : Status::Invalid(f + " needs " + std::to_string(k) + " args"); };

  if (f == "and" || f == "or") {
    Column out = BoolColumn(n);
    bool is_and = f == "and";
    for (int64_t i = 0; i < n; ++i) {
      bool acc = is_and;
      for (const auto& a : args) {
        bool v = a.IsValid(i) && a.Values<uint8_t>()[i] != 0;
        acc = is_and ? (acc && v) : (acc || v);
      }
      out.MutableValues<uint8_t>()[i] = acc;
    }
    return out;
  }
  if (f == "not") {
    ASTER_RETURN_NOT_OK(need(1));
    Column out = BoolColumn(n);
    for (int64_t i = 0; i < n; ++i) out.MutableValues<uint8_t>()[i] = !(args[0].IsValid(i) && args[0].Values<uint8_t>()[i]);
    return out;
  }
  if (f == "is_null" || f == "is_not_null") {
    ASTER_RETURN_NOT_OK(need(1));
    Column out = BoolColumn(n);
    for (int64_t i = 0; i < n; ++i) out.MutableValues<uint8_t>()[i] = (args[0].IsValid(i)) == (f == "is_not_null");
    return out;
  }
  if (f == "equal" || f == "not_equal" || f == "lt" || f == "lte" || f == "gt" || f == "gte") {
    ASTER_RETURN_NOT_OK(need(2));
    Column out = BoolColumn(n);
    std::vector<bool> valid = ValidMask(args, n);
    const Column& a = args[0]; const Column& b = args[1];
    bool str = IsStr(a.type.id) || IsStr(b.type.id);
    bool ints = !str && !IsFloat(a.type.id) && !IsFloat(b.type.id) && a.type.id != TypeId::Decimal64 && b.type.id != TypeId::Decimal64;
    bool same_dict = a.is_dictionary_encoded() && b.is_dictionary_encoded() && a.dictionary_id && a.dictionary_id == b.dictionary_id && (f == "equal" || f == "not_equal");
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      int cmp;
      if (same_dict) cmp = a.Values<int32_t>()[i] == b.Values<int32_t>()[i] ? 0 : 1;
      else if (str) cmp = CompareStr(a.GetString(i), b.GetString(i));
      else if (ints) { int64_t x = AsInt(a, i), y = AsInt(b, i); cmp = x < y ? -1 : x > y ? 1 : 0; }
      else { double x = AsDouble(a, i), y = AsDouble(b, i); cmp = x < y ? -1 : x > y ? 1 : 0; }
      bool r = f == "equal" ? cmp == 0 : f == "not_equal" ? cmp != 0 : f == "lt" ? cmp < 0 : f == "lte" ? cmp <= 0 : f == "gt" ? cmp > 0 : cmp >= 0;
      out.MutableValues<uint8_t>()[i] = r;
    }
    return WithNulls(std::move(out), valid);
  }
  if (f == "between") {
    ASTER_RETURN_NOT_OK(need(3));
    Column out = BoolColumn(n);
    std::vector<bool> valid = ValidMask(args, n);
    bool str = IsStr(args[0].type.id);
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      bool r;
      if (str) r = args[0].GetString(i) >= args[1].GetString(i) && args[0].GetString(i) <= args[2].GetString(i);
      else { double v = AsDouble(args[0], i); r = v >= AsDouble(args[1], i) && v <= AsDouble(args[2], i); }
      out.MutableValues<uint8_t>()[i] = r;
    }
    return WithNulls(std::move(out), valid);
  }
  if (f == "in") {
    ASTER_RETURN_NOT_OK(need(2));
    Column out = BoolColumn(n);
    bool str = IsStr(args[0].type.id);
    for (int64_t i = 0; i < n; ++i) {
      if (!args[0].IsValid(i)) continue;
      bool r = false;
      for (size_t k = 1; k < args.size() && !r; ++k) {
        if (!args[k].IsValid(i)) continue;
        r = str ? args[0].GetString(i) == args[k].GetString(i) : AsDouble(args[0], i) == AsDouble(args[k], i);
      }
      out.MutableValues<uint8_t>()[i] = r;
    }
    return out;
  }
  if (f == "add" || f == "subtract" || f == "multiply" || f == "divide" || f == "modulus") {
    ASTER_RETURN_NOT_OK(need(2));
    std::vector<bool> valid = ValidMask(args, n);
    const Column& a = args[0]; const Column& b = args[1];
    bool flt = IsFloat(a.type.id) || IsFloat(b.type.id) || a.type.id == TypeId::Decimal64 || b.type.id == TypeId::Decimal64 || f == "divide";
    if (flt) {
      // Decimals compute in double; precision loss is bounded by TPC-H tolerances and noted in docs.
      std::vector<double> out(n, 0);
      for (int64_t i = 0; i < n; ++i) {
        if (!valid[i]) continue;
        double x = AsDouble(a, i), y = AsDouble(b, i);
        if (f == "add") out[i] = x + y; else if (f == "subtract") out[i] = x - y; else if (f == "multiply") out[i] = x * y;
        else if (f == "divide") { if (y == 0) { valid[i] = false; } else out[i] = x / y; }
        else out[i] = std::fmod(x, y);
      }
      return WithNulls(MakeColumn<double>(TypeId::Float64, out), valid);
    }
    std::vector<int64_t> out(n, 0);
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      int64_t x = AsInt(a, i), y = AsInt(b, i);
      if (f == "add") out[i] = x + y; else if (f == "subtract") out[i] = x - y; else if (f == "multiply") out[i] = x * y;
      else { if (y == 0) { valid[i] = false; } else out[i] = x % y; }
    }
    Column c = MakeColumn<int64_t>(TypeId::Int64, out);
    if (a.type.id == TypeId::Date32 && (f == "add" || f == "subtract") && b.type.id != TypeId::Date32) {
      std::vector<int32_t> d(n); for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(out[i]);
      c = MakeColumn<int32_t>(TypeId::Date32, d);
    }
    return WithNulls(std::move(c), valid);
  }
  if (f == "negate" || f == "abs" || f == "round" || f == "floor" || f == "ceil") {
    ASTER_RETURN_NOT_OK(need(1));
    std::vector<bool> valid = ValidMask(args, n);
    const Column& a = args[0];
    if (IsFloat(a.type.id) || a.type.id == TypeId::Decimal64) {
      std::vector<double> out(n);
      for (int64_t i = 0; i < n; ++i) {
        double x = AsDouble(a, i);
        out[i] = f == "negate" ? -x : f == "abs" ? std::fabs(x) : f == "round" ? std::round(x) : f == "floor" ? std::floor(x) : std::ceil(x);
      }
      return WithNulls(MakeColumn<double>(TypeId::Float64, out), valid);
    }
    std::vector<int64_t> out(n);
    for (int64_t i = 0; i < n; ++i) { int64_t x = AsInt(a, i); out[i] = f == "negate" ? -x : f == "abs" ? std::llabs(x) : x; }
    return WithNulls(MakeColumn<int64_t>(TypeId::Int64, out), valid);
  }
  if (f == "coalesce") {
    ASTER_RETURN_NOT_OK(need(1));
    std::vector<RowIdx> pick(n, 0);
    std::vector<bool> valid(n, false);
    for (int64_t i = 0; i < n; ++i)
      for (size_t k = 0; k < args.size(); ++k) if (args[k].IsValid(i)) { pick[i] = static_cast<RowIdx>(k); valid[i] = true; break; }
    Column out = args[0];
    if (IsStr(out.type.id)) {
      std::vector<std::string> s(n);
      for (int64_t i = 0; i < n; ++i) if (valid[i]) s[i] = std::string(args[pick[i]].GetString(i));
      return WithNulls(MakeStringColumn(s), valid);
    }
    std::vector<double> d(n);
    for (int64_t i = 0; i < n; ++i) if (valid[i]) d[i] = AsDouble(args[pick[i]], i);
    if (IsFloat(out.type.id)) return WithNulls(MakeColumn<double>(TypeId::Float64, d), valid);
    std::vector<int64_t> iv(n); for (int64_t i = 0; i < n; ++i) iv[i] = static_cast<int64_t>(d[i]);
    return WithNulls(MakeColumn<int64_t>(TypeId::Int64, iv), valid);
  }
  if (f == "if_then") {
    ASTER_RETURN_NOT_OK(need(3));
    // args: cond1, then1, cond2, then2, ..., else
    size_t pairs = (args.size() - 1) / 2;
    const Column& els = args.back();
    std::vector<size_t> pick(n, args.size() - 1);
    for (int64_t i = 0; i < n; ++i)
      for (size_t p = 0; p < pairs; ++p) { const Column& c = args[p * 2]; if (c.IsValid(i) && c.Values<uint8_t>()[i]) { pick[i] = p * 2 + 1; break; } }
    std::vector<bool> valid(n);
    for (int64_t i = 0; i < n; ++i) valid[i] = args[pick[i]].IsValid(i);
    if (IsStr(els.type.id)) {
      std::vector<std::string> s(n);
      for (int64_t i = 0; i < n; ++i) if (valid[i]) s[i] = std::string(args[pick[i]].GetString(i));
      return WithNulls(MakeStringColumn(s), valid);
    }
    bool flt = false;
    for (size_t p = 0; p <= pairs; ++p) { const Column& c = args[std::min(p * 2 + 1, args.size() - 1)]; flt |= IsFloat(c.type.id) || c.type.id == TypeId::Decimal64; }
    if (flt) { std::vector<double> d(n); for (int64_t i = 0; i < n; ++i) if (valid[i]) d[i] = AsDouble(args[pick[i]], i); return WithNulls(MakeColumn<double>(TypeId::Float64, d), valid); }
    std::vector<int64_t> iv(n); for (int64_t i = 0; i < n; ++i) if (valid[i]) iv[i] = AsInt(args[pick[i]], i);
    return WithNulls(MakeColumn<int64_t>(TypeId::Int64, iv), valid);
  }
  if (f == "like" || f == "starts_with" || f == "ends_with" || f == "contains") {
    ASTER_RETURN_NOT_OK(need(2));
    Column out = BoolColumn(n);
    std::vector<bool> valid = ValidMask(args, n);
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      std::string_view s = args[0].GetString(i), p = args[1].GetString(i);
      bool r = f == "like" ? LikeMatch(s, p) : f == "starts_with" ? s.substr(0, p.size()) == p
             : f == "ends_with" ? (s.size() >= p.size() && s.substr(s.size() - p.size()) == p) : s.find(p) != std::string::npos;
      out.MutableValues<uint8_t>()[i] = r;
    }
    return WithNulls(std::move(out), valid);
  }
  if (f == "substring") {
    ASTER_RETURN_NOT_OK(need(2));
    std::vector<std::string> out(n);
    std::vector<bool> valid = ValidMask(args, n);
    for (int64_t i = 0; i < n; ++i) {
      if (!valid[i]) continue;
      std::string_view s = args[0].GetString(i);
      int64_t start = std::max<int64_t>(1, AsInt(args[1], i)) - 1;
      int64_t len = args.size() > 2 ? AsInt(args[2], i) : static_cast<int64_t>(s.size());
      if (start < static_cast<int64_t>(s.size()) && len > 0) out[i] = std::string(s.substr(start, len));
    }
    return WithNulls(MakeStringColumn(out), valid);
  }
  if (f == "length") {
    ASTER_RETURN_NOT_OK(need(1));
    std::vector<int64_t> out(n);
    for (int64_t i = 0; i < n; ++i) if (args[0].IsValid(i)) out[i] = static_cast<int64_t>(args[0].GetString(i).size());
    return WithNulls(MakeColumn<int64_t>(TypeId::Int64, out), ValidMask(args, n));
  }
  if (f == "lower" || f == "upper" || f == "trim") {
    ASTER_RETURN_NOT_OK(need(1));
    std::vector<std::string> out(n);
    for (int64_t i = 0; i < n; ++i) {
      if (!args[0].IsValid(i)) continue;
      std::string s(args[0].GetString(i));
      if (f == "lower") for (auto& ch : s) ch = static_cast<char>(tolower(ch));
      else if (f == "upper") for (auto& ch : s) ch = static_cast<char>(toupper(ch));
      else { size_t b = s.find_first_not_of(' '); size_t e2 = s.find_last_not_of(' '); s = b == std::string::npos ? "" : s.substr(b, e2 - b + 1); }
      out[i] = s;
    }
    return WithNulls(MakeStringColumn(out), ValidMask(args, n));
  }
  if (f == "concat") {
    std::vector<std::string> out(n);
    for (int64_t i = 0; i < n; ++i) for (const auto& a : args) if (a.IsValid(i)) out[i] += a.GetString(i);
    return MakeStringColumn(out);
  }
  if (f == "extract" || f == "year" || f == "month") {
    ASTER_RETURN_NOT_OK(need(1));
    const Column& date = args.back();
    std::string part = f == "year" ? "YEAR" : f == "month" ? "MONTH" : (args.size() > 1 && IsStr(args[0].type.id) && n ? std::string(args[0].GetString(0)) : "YEAR");
    std::vector<int64_t> out(n);
    std::vector<bool> valid(n);
    for (int64_t i = 0; i < n; ++i) {
      valid[i] = date.IsValid(i);
      if (!valid[i]) continue;
      int32_t days = date.type.id == TypeId::Timestamp ? static_cast<int32_t>(AsInt(date, i) / 86400000000LL) : static_cast<int32_t>(AsInt(date, i));
      out[i] = part == "MONTH" ? DaysToMonth(days) : DaysToYear(days);
    }
    return WithNulls(MakeColumn<int64_t>(TypeId::Int64, out), valid);
  }
  if (f == "hash") {
    std::vector<int64_t> out(n, 0);
    for (int64_t i = 0; i < n; ++i) {
      uint64_t h = 0;
      for (const auto& a : args) h = HashCombine(h, IsStr(a.type.id) ? Hash64(a.GetString(i)) : HashInt(static_cast<uint64_t>(AsInt(a, i))));
      out[i] = static_cast<int64_t>(h);
    }
    return MakeColumn<int64_t>(TypeId::Int64, out);
  }
  if (f == "vector_distance") {
    ASTER_RETURN_NOT_OK(need(2));
    const Column& a = args[0]; const Column& b = args[1];
    if (a.type.id != TypeId::FixedVector || b.type.id != TypeId::FixedVector || a.type.width != b.type.width)
      return Status::Invalid("vector_distance needs two vectors of equal dims");
    uint32_t d = a.type.width;
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; ++i) {
      const float* x = a.Values<float>() + i * d; const float* y = b.Values<float>() + i * d;
      double s = 0; for (uint32_t k = 0; k < d; ++k) { double t = double(x[k]) - y[k]; s += t * t; }
      out[i] = static_cast<float>(std::sqrt(s));
    }
    return WithNulls(MakeColumn<float>(TypeId::Float32, out), ValidMask(args, n));
  }
  return Status::NotSupported("function " + f);
}

Result<SelectionVector> ExpressionEvaluator::Filter(const Expr& pred, const RecordBatch& in, const SelectionVector& sel) {
  ASTER_ASSIGN_OR_RETURN(Column mask, Evaluate(pred, in));
  if (mask.type.id != TypeId::Bool) return Status::Invalid("predicate is not boolean");
  const uint8_t* v = mask.Values<uint8_t>();
  std::vector<RowIdx> rows;
  if (sel.all) {
    rows.reserve(in.num_rows());
    for (int64_t i = 0; i < in.num_rows(); ++i) if (v[i] && mask.IsValid(i)) rows.push_back(static_cast<RowIdx>(i));
  } else {
    rows.reserve(sel.rows.size());
    for (RowIdx r : sel.rows) if (v[r] && mask.IsValid(r)) rows.push_back(r);
  }
  return SelectionVector::Of(std::move(rows));
}

}  // namespace aster::exec
