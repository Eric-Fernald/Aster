#pragma once
#include "aster/common/column.hpp"
#include "aster/storage/encoding.hpp"

namespace aster::storage {

struct ColumnProfile {
  uint32_t rows = 0;
  uint32_t nulls = 0;
  uint32_t distinct_estimate = 0;
  double avg_run_length = 1.0;
  bool monotonic = false;
  bool sorted = false;
  int64_t min_i64 = 0, max_i64 = 0;
  uint8_t for_bits = 64;
  uint8_t delta_bits = 64;
};

ColumnProfile ProfileColumn(const Column& col);

// Lightweight encodings are the primary representation. Selection follows the plan's priority:
// dictionary for low cardinality, FOR for narrow integer ranges, RLE for clustered, delta for monotonic.
Encoding SelectEncoding(const Column& col, const ColumnProfile& profile);
Encoding SelectEncoding(const Column& col);

}  // namespace aster::storage
