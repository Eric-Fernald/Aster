#pragma once
#include <atomic>
#include <chrono>
#include <string>

#include "aster/common/status.hpp"
#include "aster/ingest/compactor.hpp"
#include "aster/ingest/delta_store.hpp"
#include "aster/ingest/wal.hpp"

namespace aster::ingest {

struct IngestStats {
  uint64_t appends = 0;
  uint64_t rows = 0;
  uint64_t bytes = 0;
  double seconds = 0;
  double bytes_per_sec() const { return seconds > 0 ? bytes / seconds : 0; }
};

// Append path: WAL to NVMe with fsync, then the delta store, then acknowledge. Queries keep running;
// the compactor is nudged when the delta crosses its budget.
class IngestService {
 public:
  IngestService(WriteAheadLog& wal, DeltaStore& delta, Compactor* compactor);
  Result<uint64_t> Append(const std::string& table, RecordBatchPtr batch);
  IngestStats stats() const;
  void SetFaultHook(std::function<void(const char* point)> hook) { fault_hook_ = std::move(hook); }

 private:
  WriteAheadLog& wal_;
  DeltaStore& delta_;
  Compactor* compactor_;
  std::atomic<uint64_t> appends_{0}, rows_{0}, bytes_{0};
  std::atomic<double> seconds_{0};
  std::function<void(const char*)> fault_hook_;
};

}  // namespace aster::ingest
