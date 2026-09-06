#pragma once
#include <string>

#include "aster/common/status.hpp"
#include "aster/ingest/delta_store.hpp"
#include "aster/ingest/wal.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::durability {

struct RecoveryReport {
  uint64_t wal_records = 0;
  uint64_t wal_rows = 0;
  uint64_t wal_bytes = 0;
  uint64_t segments_loaded = 0;
  uint64_t segments_rejected = 0;
  uint64_t tables = 0;
  double seconds = 0;
  std::string ToString() const;
};

// Nothing in VRAM is ever the only copy. Recovery reloads the segment lists, verifies footers,
// replays the WAL into the delta store, and leaves VRAM to refill on demand.
class Recovery {
 public:
  static Result<RecoveryReport> Run(storage::Catalog& catalog, ingest::WriteAheadLog& wal, ingest::DeltaStore& delta);
  static Status VerifySegments(storage::Catalog& catalog, RecoveryReport* report);
};

}  // namespace aster::durability
