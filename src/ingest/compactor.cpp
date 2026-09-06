#include "aster/ingest/compactor.hpp"

#include <chrono>

#include "aster/common/log.hpp"
#include "aster/durability/fault_injector.hpp"
#include "aster/storage/segment.hpp"

namespace aster::ingest {

Compactor::Compactor(const EngineConfig& cfg, storage::Catalog& catalog, DeltaStore& delta, WriteAheadLog& wal)
    : cfg_(cfg), catalog_(catalog), delta_(delta), wal_(wal) {}

Compactor::~Compactor() { Stop(); }

void Compactor::Start() {
  std::lock_guard<std::mutex> lk(mu_);
  if (worker_.joinable()) return;
  stop_ = false;
  worker_ = std::thread([this] { Loop(); });
}

void Compactor::Stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void Compactor::Trigger() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    triggered_ = true;
  }
  cv_.notify_all();
}

void Compactor::Loop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait_for(lk, std::chrono::seconds(1), [this] { return stop_ || triggered_; });
      if (stop_) return;
      triggered_ = false;
    }
    if (delta_.bytes() == 0) continue;
    Status s = CompactAll();
    if (!s.ok()) ASTER_LOG(Warn, "compaction failed: %s", s.ToString().c_str());
  }
}

Status Compactor::CompactAll() {
  for (const auto& t : delta_.Tables()) ASTER_RETURN_NOT_OK(CompactNow(t));
  return Status::OK();
}

Status Compactor::CompactNow(const std::string& table) {
  auto t0 = std::chrono::steady_clock::now();
  DeltaSnapshot snap = delta_.Snapshot(table);
  if (snap.batches.empty()) return Status::OK();
  if (!catalog_.HasTable(table)) return Status::NotFound("table " + table);
  ASTER_ASSIGN_OR_RETURN(auto info, catalog_.GetTable(table));

  // Pack delta batches into segments near the target size, one segment per group.
  std::vector<std::shared_ptr<storage::SegmentMeta>> added;
  std::vector<RecordBatchPtr> group;
  size_t group_bytes = 0;
  uint64_t rows = 0, bytes = 0;
  auto flush = [&]() -> Status {
    if (group.empty()) return Status::OK();
    RecordBatchPtr merged = ConcatBatches(group);
    merged->schema = info.schema;
    SegmentId id = catalog_.NextSegmentId();
    storage::SegmentWriteOptions opts;
    opts.tile_rows = cfg_.tile_rows;
    if (fault_hook_) fault_hook_("compaction.before_write");
    ASTER_FAULT_POINT("compaction.before_write");
    ASTER_ASSIGN_OR_RETURN(auto meta, storage::SegmentWriter::Write(*merged, id, catalog_.SegmentPath(table, id), table, opts));
    ASTER_FAULT_POINT("compaction.after_write");
    rows += meta.num_rows;
    bytes += meta.encoded_bytes;
    added.push_back(std::make_shared<storage::SegmentMeta>(std::move(meta)));
    group.clear();
    group_bytes = 0;
    return Status::OK();
  };
  for (const auto& b : snap.batches) {
    group.push_back(b);
    group_bytes += b->nbytes();
    if (group_bytes >= cfg_.segment_target_bytes) ASTER_RETURN_NOT_OK(flush());
  }
  ASTER_RETURN_NOT_OK(flush());

  // Atomic swap: old segment lists survive for in flight queries, then the delta and WAL shrink.
  ASTER_FAULT_POINT("compaction.before_swap");
  ASTER_RETURN_NOT_OK(catalog_.ReplaceSegments(table, {}, added));
  delta_.Release(table, snap.last_lsn);
  ASTER_FAULT_POINT("compaction.after_swap");
  uint64_t min_pending = UINT64_MAX;
  for (const auto& t : delta_.Tables()) { DeltaSnapshot s = delta_.Snapshot(t); if (!s.batches.empty()) min_pending = std::min(min_pending, s.last_lsn); }
  uint64_t truncate_to = min_pending == UINT64_MAX ? wal_.last_lsn() : std::min(snap.last_lsn, min_pending);
  Status ts = wal_.Truncate(truncate_to);
  if (!ts.ok()) ASTER_LOG(Warn, "wal truncate: %s", ts.ToString().c_str());

  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.runs++;
  stats_.segments_written += added.size();
  stats_.rows_compacted += rows;
  stats_.bytes_written += bytes;
  stats_.last_run_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ASTER_LOG(Info, "compacted %s: %zu segments, %llu rows, %.1f ms", table.c_str(), added.size(), (unsigned long long)rows, stats_.last_run_ms);
  return Status::OK();
}

CompactionStats Compactor::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

}  // namespace aster::ingest
