#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "aster/common/config.hpp"
#include "aster/common/status.hpp"
#include "aster/ingest/delta_store.hpp"
#include "aster/ingest/wal.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::ingest {

struct CompactionStats {
  uint64_t runs = 0;
  uint64_t segments_written = 0;
  uint64_t rows_compacted = 0;
  uint64_t bytes_written = 0;
  double last_run_ms = 0;
};

// Background thread: encodes delta batches into native segments (encoding selection, zone maps,
// bloom filters), writes them to NVMe, atomically swaps the segment list, then trims the WAL.
class Compactor {
 public:
  Compactor(const EngineConfig& cfg, storage::Catalog& catalog, DeltaStore& delta, WriteAheadLog& wal);
  ~Compactor();
  void Start();
  void Stop();
  void Trigger();
  Status CompactNow(const std::string& table);
  Status CompactAll();
  CompactionStats stats() const;
  void SetFaultHook(std::function<void(const char* point)> hook) { fault_hook_ = std::move(hook); }

 private:
  void Loop();
  const EngineConfig& cfg_;
  storage::Catalog& catalog_;
  DeltaStore& delta_;
  WriteAheadLog& wal_;
  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool triggered_ = false;
  mutable std::mutex stats_mu_;
  CompactionStats stats_;
  std::function<void(const char*)> fault_hook_;
};

}  // namespace aster::ingest
