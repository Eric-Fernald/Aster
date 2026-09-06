#pragma once
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"

namespace aster::storage {

struct ParquetReadOptions {
  std::vector<std::string> columns;  // empty = all
  int64_t batch_rows = 1 << 20;
};

// Parquet is an import path. With Arrow available the file is decoded through parquet::arrow;
// otherwise the reader reports NotSupported so the host system can supply Arrow batches instead.
class ParquetReader {
 public:
  static bool Available();
  static Result<Schema> ReadSchema(const std::string& path);
  static Result<std::vector<RecordBatchPtr>> ReadFile(const std::string& path, const ParquetReadOptions& opts = {});
  static Result<int64_t> RowCount(const std::string& path);
};

}  // namespace aster::storage
