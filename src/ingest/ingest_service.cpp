#include "aster/ingest/ingest_service.hpp"

#include <chrono>

#include "aster/durability/fault_injector.hpp"

namespace aster::ingest {

IngestService::IngestService(WriteAheadLog& wal, DeltaStore& delta, Compactor* compactor)
    : wal_(wal), delta_(delta), compactor_(compactor) {}

Result<uint64_t> IngestService::Append(const std::string& table, RecordBatchPtr batch) {
  if (!batch || batch->num_rows() == 0) return Status::Invalid("empty batch");
  auto t0 = std::chrono::steady_clock::now();
  if (fault_hook_) fault_hook_("ingest.before_wal");
  ASTER_ASSIGN_OR_RETURN(uint64_t lsn, wal_.Append(table, *batch, true));
  ASTER_FAULT_POINT("ingest.after_wal");
  ASTER_RETURN_NOT_OK(delta_.Append(table, batch, lsn));
  appends_++;
  rows_ += batch->num_rows();
  bytes_ += batch->nbytes();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double cur = seconds_.load();
  while (!seconds_.compare_exchange_weak(cur, cur + s)) {}
  if (compactor_ && delta_.over_budget()) compactor_->Trigger();
  return lsn;
}

IngestStats IngestService::stats() const {
  IngestStats s;
  s.appends = appends_;
  s.rows = rows_;
  s.bytes = bytes_;
  s.seconds = seconds_;
  return s;
}

}  // namespace aster::ingest
