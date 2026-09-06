#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "aster/common/column.hpp"

namespace aster::storage {

// Per column chunk statistics. Integer-like types keep int64 bounds, floats keep double, strings keep
// a 64 byte prefix. Distinct count is a HyperLogLog estimate.
struct ZoneMap {
  bool has_bounds = false;
  int64_t min_i64 = 0, max_i64 = 0;
  double min_f64 = 0, max_f64 = 0;
  std::string min_str, max_str;
  uint32_t null_count = 0;
  uint32_t row_count = 0;
  uint32_t distinct_estimate = 0;
  bool sorted_asc = false;

  static ZoneMap Compute(const Column& col);
  bool MayContainI64(int64_t v) const { return !has_bounds || (v >= min_i64 && v <= max_i64); }
  bool MayOverlapI64(int64_t lo, int64_t hi) const { return !has_bounds || !(hi < min_i64 || lo > max_i64); }
  bool MayContainF64(double v) const { return !has_bounds || (v >= min_f64 && v <= max_f64); }
  bool MayContainStr(std::string_view v) const;
  bool MayOverlapStr(std::string_view lo, std::string_view hi) const;
  void Serialize(std::vector<uint8_t>& out) const;
  static ZoneMap Deserialize(const uint8_t*& p, const uint8_t* end);
  std::string ToString() const;
};

class HyperLogLog {
 public:
  explicit HyperLogLog(uint8_t precision = 10);
  void Add(uint64_t hash);
  uint64_t Estimate() const;
  void Merge(const HyperLogLog& o);

 private:
  uint8_t p_;
  std::vector<uint8_t> reg_;
};

}  // namespace aster::storage
