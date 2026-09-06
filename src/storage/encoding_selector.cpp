#include "aster/storage/encoding_selector.hpp"

#include "aster/storage/zone_map.hpp"
#include "internal.hpp"

namespace aster::storage {

using namespace internal;

ColumnProfile ProfileColumn(const Column& col) {
  ColumnProfile p;
  p.rows = static_cast<uint32_t>(col.length);
  p.nulls = static_cast<uint32_t>(col.null_count);
  ZoneMap z = ZoneMap::Compute(col);
  p.distinct_estimate = z.distinct_estimate;
  p.sorted = z.sorted_asc;
  p.min_i64 = z.min_i64;
  p.max_i64 = z.max_i64;
  const TypeId t = col.type.id;
  if (IsIntegerLike(t) || t == TypeId::Bool) {
    uint32_t runs = 0;
    int64_t prev = 0, min_d = 0, max_d = 0;
    bool mono = true;
    for (int64_t i = 0; i < col.length; ++i) {
      int64_t v = col.IsValid(i) ? ReadI64(col, i) : prev;
      if (i == 0 || v != prev) ++runs;
      if (i > 0) {
        int64_t d = v - prev;
        if (d < 0) mono = false;
        if (i == 1) { min_d = max_d = d; } else { min_d = std::min(min_d, d); max_d = std::max(max_d, d); }
      }
      prev = v;
    }
    p.avg_run_length = runs ? double(col.length) / runs : 1.0;
    p.monotonic = mono && col.length > 1;
    p.for_bits = z.has_bounds ? BitsNeeded(static_cast<uint64_t>(z.max_i64 - z.min_i64)) : 64;
    p.delta_bits = col.length > 1 ? BitsNeeded(static_cast<uint64_t>(max_d - min_d)) : 0;
  } else if (t == TypeId::String) {
    uint32_t runs = 0;
    std::string_view prev;
    for (int64_t i = 0; i < col.length; ++i) {
      std::string_view v = col.IsValid(i) ? col.GetString(i) : prev;
      if (i == 0 || v != prev) ++runs;
      prev = v;
    }
    p.avg_run_length = runs ? double(col.length) / runs : 1.0;
  }
  return p;
}

Encoding SelectEncoding(const Column& col, const ColumnProfile& p) {
  const TypeId t = col.type.id;
  if (p.rows == 0) return Encoding::Plain;
  double distinct_ratio = double(p.distinct_estimate) / double(p.rows);
  if (t == TypeId::String) return distinct_ratio < 0.5 ? Encoding::Dictionary : Encoding::Plain;
  if (IsIntegerLike(t) || t == TypeId::Bool) {
    uint8_t src_bits = static_cast<uint8_t>(TypeByteWidth(col.type) * 8);
    if (p.avg_run_length >= 4.0) return Encoding::RunLength;
    if (p.monotonic && p.delta_bits < p.for_bits && p.delta_bits < src_bits) return Encoding::Delta;
    if (p.for_bits < src_bits) return Encoding::ForBitPack;
    if (distinct_ratio < 0.05) return Encoding::Dictionary;
    return Encoding::Plain;
  }
  return Encoding::Plain;
}

Encoding SelectEncoding(const Column& col) { return SelectEncoding(col, ProfileColumn(col)); }

}  // namespace aster::storage
