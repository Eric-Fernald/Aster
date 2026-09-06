#include "aster/durability/recovery.hpp"

#include <chrono>
#include <sstream>

#include "aster/common/log.hpp"

namespace aster::durability {

std::string RecoveryReport::ToString() const {
  std::ostringstream os;
  os << "recovery: tables=" << tables << " segments=" << segments_loaded << " rejected=" << segments_rejected
     << " wal_records=" << wal_records << " wal_rows=" << wal_rows << " wal_bytes=" << wal_bytes << " seconds=" << seconds;
  return os.str();
}

Status Recovery::VerifySegments(storage::Catalog& catalog, RecoveryReport* report) {
  for (const auto& table : catalog.Tables()) {
    ASTER_ASSIGN_OR_RETURN(auto snap, catalog.Snapshot(table));
    std::vector<std::shared_ptr<storage::SegmentMeta>> good;
    bool changed = false;
    for (const auto& seg : snap->segments) {
      // The footer checksum was verified at manifest load; re-read it to reject torn segment files.
      auto meta = storage::SegmentReader::ReadMeta(seg->path);
      if (meta.ok()) { good.push_back(seg); if (report) ++report->segments_loaded; }
      else { changed = true; if (report) ++report->segments_rejected; ASTER_LOG(Warn, "rejecting segment %s: %s", seg->path.c_str(), meta.status().ToString().c_str()); }
    }
    if (changed) ASTER_RETURN_NOT_OK(catalog.SwapSegments(table, good));
    if (report) ++report->tables;
  }
  return Status::OK();
}

Result<RecoveryReport> Recovery::Run(storage::Catalog& catalog, ingest::WriteAheadLog& wal, ingest::DeltaStore& delta) {
  RecoveryReport r;
  auto t0 = std::chrono::steady_clock::now();
  ASTER_RETURN_NOT_OK(catalog.Open());
  ASTER_RETURN_NOT_OK(VerifySegments(catalog, &r));
  ASTER_RETURN_NOT_OK(wal.Open());
  ASTER_RETURN_NOT_OK(wal.Replay([&](const ingest::WriteAheadLog::Record& rec) {
    if (!catalog.HasTable(rec.table)) { ASTER_LOG(Warn, "wal record for unknown table %s skipped", rec.table.c_str()); return Status::OK(); }
    ++r.wal_records;
    r.wal_rows += rec.batch->num_rows();
    r.wal_bytes += rec.batch->nbytes();
    return delta.Append(rec.table, rec.batch, rec.lsn);
  }));
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASTER_LOG(Info, "%s", r.ToString().c_str());
  return r;
}

}  // namespace aster::durability
