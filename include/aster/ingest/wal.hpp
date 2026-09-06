#pragma once
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"

namespace aster::ingest {

// Append only log on NVMe. Every record is length prefixed and CRC32C protected; fsync happens
// before the append is acknowledged. A truncated tail from a crash mid write is detected and dropped.
class WriteAheadLog {
 public:
  struct Record {
    uint64_t lsn = 0;
    std::string table;
    RecordBatchPtr batch;
  };

  explicit WriteAheadLog(std::string dir);
  Status Open();
  Result<uint64_t> Append(const std::string& table, const RecordBatch& batch, bool fsync = true);
  Status Replay(const std::function<Status(const Record&)>& fn);
  Status Truncate(uint64_t up_to_lsn);  // records at or below are durable elsewhere
  uint64_t last_lsn() const { return last_lsn_; }
  uint64_t bytes() const { return bytes_; }
  const std::string& path() const { return path_; }

  static std::vector<uint8_t> SerializeBatch(const std::string& table, const RecordBatch& batch);
  static Result<Record> DeserializeBatch(const uint8_t* data, size_t len);

 private:
  std::string dir_, path_;
  std::mutex mu_;
  uint64_t last_lsn_ = 0;
  uint64_t bytes_ = 0;
};

}  // namespace aster::ingest
