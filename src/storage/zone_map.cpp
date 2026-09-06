#include "aster/storage/zone_map.hpp"

#include <cmath>
#include <sstream>

#include "aster/common/hash.hpp"
#include "internal.hpp"

namespace aster::storage {

using namespace internal;

HyperLogLog::HyperLogLog(uint8_t precision) : p_(precision), reg_(size_t(1) << precision, 0) {}

void HyperLogLog::Add(uint64_t hash) {
  size_t idx = hash >> (64 - p_);
  uint64_t rest = (hash << p_) | (1ULL << (p_ - 1));
  uint8_t rank = 1;
  while (!(rest & (1ULL << 63))) { rest <<= 1; ++rank; }
  if (rank > reg_[idx]) reg_[idx] = rank;
}

uint64_t HyperLogLog::Estimate() const {
  double m = double(reg_.size());
  double sum = 0;
  int zeros = 0;
  for (uint8_t r : reg_) { sum += std::ldexp(1.0, -r); if (!r) ++zeros; }
  double alpha = 0.7213 / (1.0 + 1.079 / m);
  double e = alpha * m * m / sum;
  if (e <= 2.5 * m && zeros) e = m * std::log(m / zeros);
  return static_cast<uint64_t>(e + 0.5);
}

void HyperLogLog::Merge(const HyperLogLog& o) {
  for (size_t i = 0; i < reg_.size(); ++i) reg_[i] = std::max(reg_[i], o.reg_[i]);
}

ZoneMap ZoneMap::Compute(const Column& col) {
  ZoneMap z;
  z.row_count = static_cast<uint32_t>(col.length);
  z.null_count = static_cast<uint32_t>(col.null_count);
  HyperLogLog hll;
  bool first = true;
  z.sorted_asc = true;
  const TypeId t = col.type.id;
  if (IsIntegerLike(t) || t == TypeId::Bool) {
    int64_t prev = 0;
    for (int64_t i = 0; i < col.length; ++i) {
      if (!col.IsValid(i)) continue;
      int64_t v = ReadI64(col, i);
      hll.Add(HashInt(static_cast<uint64_t>(v)));
      if (first) { z.min_i64 = z.max_i64 = v; first = false; }
      else { z.min_i64 = std::min(z.min_i64, v); z.max_i64 = std::max(z.max_i64, v); if (v < prev) z.sorted_asc = false; }
      prev = v;
    }
    z.min_f64 = double(z.min_i64); z.max_f64 = double(z.max_i64);
  } else if (t == TypeId::Float32 || t == TypeId::Float64) {
    double prev = 0;
    for (int64_t i = 0; i < col.length; ++i) {
      if (!col.IsValid(i)) continue;
      double v = t == TypeId::Float32 ? col.Values<float>()[i] : col.Values<double>()[i];
      uint64_t bits; std::memcpy(&bits, &v, 8);
      hll.Add(HashInt(bits));
      if (first) { z.min_f64 = z.max_f64 = v; first = false; }
      else { z.min_f64 = std::min(z.min_f64, v); z.max_f64 = std::max(z.max_f64, v); if (v < prev) z.sorted_asc = false; }
      prev = v;
    }
  } else if (t == TypeId::String || t == TypeId::Binary) {
    std::string_view prev;
    for (int64_t i = 0; i < col.length; ++i) {
      if (!col.IsValid(i)) continue;
      std::string_view v = col.GetString(i);
      hll.Add(Hash64(v));
      if (first) { z.min_str = z.max_str = std::string(v.substr(0, 64)); first = false; }
      else {
        if (v < z.min_str) z.min_str = std::string(v.substr(0, 64));
        if (v > z.max_str) z.max_str = std::string(v.substr(0, 64));
        if (v < prev) z.sorted_asc = false;
      }
      prev = v;
    }
  } else {
    z.sorted_asc = false;
  }
  z.has_bounds = !first;
  z.distinct_estimate = static_cast<uint32_t>(std::min<uint64_t>(hll.Estimate(), col.length));
  return z;
}

bool ZoneMap::MayContainStr(std::string_view v) const {
  if (!has_bounds) return true;
  // max_str is a 64 byte prefix so a value longer than that may sort after it.
  return v >= min_str && (v.substr(0, 64) <= max_str || max_str.size() == 64);
}

bool ZoneMap::MayOverlapStr(std::string_view lo, std::string_view hi) const {
  if (!has_bounds) return true;
  return !(hi < min_str || (lo.substr(0, 64) > max_str && max_str.size() < 64));
}

void ZoneMap::Serialize(std::vector<uint8_t>& out) const {
  PutVal<uint8_t>(out, has_bounds);
  PutVal<uint8_t>(out, sorted_asc);
  PutVal<int64_t>(out, min_i64); PutVal<int64_t>(out, max_i64);
  PutVal<double>(out, min_f64); PutVal<double>(out, max_f64);
  PutVal<uint32_t>(out, null_count); PutVal<uint32_t>(out, row_count); PutVal<uint32_t>(out, distinct_estimate);
  PutVal<uint16_t>(out, static_cast<uint16_t>(min_str.size())); Put(out, min_str.data(), min_str.size());
  PutVal<uint16_t>(out, static_cast<uint16_t>(max_str.size())); Put(out, max_str.data(), max_str.size());
}

ZoneMap ZoneMap::Deserialize(const uint8_t*& p, const uint8_t*) {
  ZoneMap z;
  z.has_bounds = GetVal<uint8_t>(p);
  z.sorted_asc = GetVal<uint8_t>(p);
  z.min_i64 = GetVal<int64_t>(p); z.max_i64 = GetVal<int64_t>(p);
  z.min_f64 = GetVal<double>(p); z.max_f64 = GetVal<double>(p);
  z.null_count = GetVal<uint32_t>(p); z.row_count = GetVal<uint32_t>(p); z.distinct_estimate = GetVal<uint32_t>(p);
  uint16_t n = GetVal<uint16_t>(p); z.min_str.assign(reinterpret_cast<const char*>(p), n); p += n;
  n = GetVal<uint16_t>(p); z.max_str.assign(reinterpret_cast<const char*>(p), n); p += n;
  return z;
}

std::string ZoneMap::ToString() const {
  std::ostringstream os;
  os << "rows=" << row_count << " nulls=" << null_count << " distinct~" << distinct_estimate;
  if (has_bounds) os << " i64[" << min_i64 << "," << max_i64 << "] f64[" << min_f64 << "," << max_f64 << "]";
  if (!min_str.empty() || !max_str.empty()) os << " str[" << min_str << "," << max_str << "]";
  if (sorted_asc) os << " sorted";
  return os.str();
}

}  // namespace aster::storage
