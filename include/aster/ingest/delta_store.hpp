#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"

namespace aster::ingest {

struct DeltaSnapshot {
  std::vector<RecordBatchPtr> batches;
  uint64_t rows = 0;
  uint64_t bytes = 0;
  uint64_t last_lsn = 0;
};

// CPU side delta store: Arrow batches in pinned host memory, unioned with base segments at query
// time. The compactor drains it into native segments and the memory stays under a fixed budget.
class DeltaStore {
 public:
  DeltaStore(hal::DevicePtr dev, size_t max_bytes);
  Status Append(const std::string& table, RecordBatchPtr batch, uint64_t lsn);
  DeltaSnapshot Snapshot(const std::string& table) const;
  // Removes every batch with lsn <= up_to (they now live in segments).
  void Release(const std::string& table, uint64_t up_to_lsn);
  uint64_t Rows(const std::string& table) const;
  size_t bytes() const;
  size_t max_bytes() const { return max_bytes_; }
  bool over_budget() const { return bytes() > max_bytes_; }
  std::vector<std::string> Tables() const;

 private:
  struct Entry { RecordBatchPtr batch; uint64_t lsn; };
  RecordBatchPtr ToPinned(const RecordBatch& b) const;
  hal::DevicePtr dev_;
  size_t max_bytes_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<Entry>> tables_;
  size_t bytes_ = 0;
};

}  // namespace aster::ingest
