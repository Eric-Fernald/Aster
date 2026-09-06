#include <fstream>

#include "aster/durability/fault_injector.hpp"
#include "aster/durability/recovery.hpp"
#include "aster/ingest/compactor.hpp"
#include "aster/ingest/delta_store.hpp"
#include "aster/ingest/ingest_service.hpp"
#include "aster/ingest/wal.hpp"
#include "aster/memory/gds.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::ingest;

ASTER_TEST(wal_append_replay_truncate) {
  std::string dir = aster_test::TempDir("wal");
  WriteAheadLog wal(dir);
  ASTER_CHECK_OK(wal.Open());
  auto b = aster_test::MakeLineitem(100);
  ASTER_ASSIGN_OK(uint64_t l1, wal.Append("lineitem", *b));
  ASTER_ASSIGN_OK(uint64_t l2, wal.Append("lineitem", *b));
  ASTER_CHECK_EQ(l1, 1u);
  ASTER_CHECK_EQ(l2, 2u);
  int seen = 0;
  WriteAheadLog again(dir);
  ASTER_CHECK_OK(again.Open());
  ASTER_CHECK_OK(again.Replay([&](const WriteAheadLog::Record& r) {
    ++seen;
    if (r.batch->num_rows() != 100 || r.table != "lineitem") return Status::Corrupt("bad record");
    if (std::string(r.batch->columns[5].GetString(3)) != std::string(b->columns[5].GetString(3))) return Status::Corrupt("string mismatch");
    return Status::OK();
  }));
  ASTER_CHECK_EQ(seen, 2);
  ASTER_CHECK_EQ(again.last_lsn(), 2u);
  ASTER_CHECK_OK(again.Truncate(1));
  seen = 0;
  ASTER_CHECK_OK(again.Replay([&](const WriteAheadLog::Record& r) { ++seen; return r.lsn == 2 ? Status::OK() : Status::Corrupt("lsn"); }));
  ASTER_CHECK_EQ(seen, 1);
}

ASTER_TEST(wal_torn_tail_is_dropped) {
  std::string dir = aster_test::TempDir("torn");
  WriteAheadLog wal(dir);
  ASTER_CHECK_OK(wal.Open());
  auto b = aster_test::MakeLineitem(50);
  ASTER_CHECK_OK(wal.Append("t", *b).status());
  ASTER_CHECK_OK(wal.Append("t", *b).status());
  ASTER_ASSIGN_OK(auto bytes, memory::ReadWholeFile(wal.path()));
  std::vector<uint8_t> torn(bytes.begin(), bytes.begin() + bytes.size() - 37);
  ASTER_CHECK_OK(memory::WriteFile(wal.path(), torn.data(), torn.size(), true));
  WriteAheadLog re(dir);
  ASTER_CHECK_OK(re.Open());
  int seen = 0;
  ASTER_CHECK_OK(re.Replay([&](const WriteAheadLog::Record&) { ++seen; return Status::OK(); }));
  ASTER_CHECK_EQ(seen, 1);
  ASTER_CHECK_OK(re.Append("t", *b).status());
  ASTER_CHECK_EQ(re.last_lsn(), 2u);
}

ASTER_TEST(delta_store_snapshot_and_release) {
  DeltaStore d(hal::CpuDevice(), 1 << 20);
  auto b = aster_test::MakeLineitem(10);
  ASTER_CHECK_OK(d.Append("t", b, 1));
  ASTER_CHECK_OK(d.Append("t", b, 2));
  ASTER_CHECK_EQ(d.Rows("t"), 20u);
  DeltaSnapshot s = d.Snapshot("t");
  ASTER_CHECK_EQ(s.batches.size(), size_t(2));
  ASTER_CHECK_EQ(s.last_lsn, 2u);
  d.Release("t", 1);
  ASTER_CHECK_EQ(d.Rows("t"), 10u);
  ASTER_CHECK(d.bytes() > 0);
}

ASTER_TEST(ingest_compact_recover_cycle) {
  std::string root = aster_test::TempDir("ingest");
  EngineConfig cfg;
  cfg.data_dir = root + "/data";
  cfg.wal_dir = root + "/wal";
  cfg.segment_target_bytes = 1 << 16;
  storage::Catalog catalog(cfg.data_dir);
  ASTER_CHECK_OK(catalog.Open());
  storage::TableInfo info;
  info.name = "lineitem";
  info.schema = aster_test::LineitemSchema();
  ASTER_CHECK_OK(catalog.CreateTable(info));
  WriteAheadLog wal(cfg.wal_dir);
  ASTER_CHECK_OK(wal.Open());
  DeltaStore delta(hal::CpuDevice(), 1 << 30);
  Compactor compactor(cfg, catalog, delta, wal);
  IngestService ingest(wal, delta, &compactor);
  for (int i = 0; i < 5; ++i) ASTER_CHECK_OK(ingest.Append("lineitem", aster_test::MakeLineitem(2000, i)).status());
  ASTER_CHECK_EQ(ingest.stats().rows, 10000u);
  ASTER_CHECK(ingest.stats().bytes_per_sec() > 0);
  ASTER_CHECK_EQ(delta.Rows("lineitem"), 10000u);

  // Unflushed rows survive a crash through the WAL.
  {
    storage::Catalog cat2(cfg.data_dir);
    WriteAheadLog wal2(cfg.wal_dir);
    DeltaStore delta2(hal::CpuDevice(), 1 << 30);
    ASTER_ASSIGN_OK(auto report, durability::Recovery::Run(cat2, wal2, delta2));
    ASTER_CHECK_EQ(report.wal_records, 5u);
    ASTER_CHECK_EQ(delta2.Rows("lineitem"), 10000u);
  }

  ASTER_CHECK_OK(compactor.CompactNow("lineitem"));
  ASTER_CHECK_EQ(delta.Rows("lineitem"), 0u);
  ASTER_ASSIGN_OK(auto snap, catalog.Snapshot("lineitem"));
  ASTER_CHECK(snap->segments.size() >= 2);
  ASTER_CHECK_EQ(snap->total_rows(), 10000u);
  ASTER_CHECK(wal.bytes() == 0);
  CompactionStats cs = compactor.stats();
  ASTER_CHECK_EQ(cs.rows_compacted, 10000u);

  // After compaction the WAL is empty and recovery loads only segments.
  storage::Catalog cat3(cfg.data_dir);
  WriteAheadLog wal3(cfg.wal_dir);
  DeltaStore delta3(hal::CpuDevice(), 1 << 30);
  ASTER_ASSIGN_OK(auto report3, durability::Recovery::Run(cat3, wal3, delta3));
  ASTER_CHECK_EQ(report3.wal_records, 0u);
  ASTER_CHECK_EQ(report3.segments_loaded, snap->segments.size());
  ASTER_CHECK_EQ(delta3.Rows("lineitem"), 0u);
}

ASTER_TEST(chaos_crash_mid_compaction_keeps_data) {
  std::string root = aster_test::TempDir("chaos");
  EngineConfig cfg;
  cfg.data_dir = root + "/data";
  cfg.wal_dir = root + "/wal";
  storage::Catalog catalog(cfg.data_dir);
  ASTER_CHECK_OK(catalog.Open());
  storage::TableInfo info;
  info.name = "t";
  info.schema = aster_test::LineitemSchema();
  ASTER_CHECK_OK(catalog.CreateTable(info));
  WriteAheadLog wal(cfg.wal_dir);
  ASTER_CHECK_OK(wal.Open());
  DeltaStore delta(hal::CpuDevice(), 1 << 30);
  Compactor compactor(cfg, catalog, delta, wal);
  IngestService ingest(wal, delta, nullptr);
  ASTER_CHECK_OK(ingest.Append("t", aster_test::MakeLineitem(1000)).status());
  // Crash after the segment is written but before the catalog swap: the WAL still holds the rows.
  auto& fi = durability::FaultInjector::Global();
  fi.Arm("compaction.before_swap", [] { throw std::runtime_error("simulated crash"); });
  bool threw = false;
  try { (void)compactor.CompactNow("t"); } catch (const std::runtime_error&) { threw = true; }
  fi.Reset();
  ASTER_CHECK(threw);
  storage::Catalog cat2(cfg.data_dir);
  WriteAheadLog wal2(cfg.wal_dir);
  DeltaStore delta2(hal::CpuDevice(), 1 << 30);
  ASTER_ASSIGN_OK(auto report, durability::Recovery::Run(cat2, wal2, delta2));
  ASTER_CHECK_EQ(delta2.Rows("t"), 1000u);
  ASTER_ASSIGN_OK(auto snap, cat2.Snapshot("t"));
  ASTER_CHECK_EQ(snap->segments.size(), size_t(0));  // orphan file is not in the manifest
  // Crash mid WAL write: the torn record is discarded, earlier rows remain.
  fi.Arm("wal.after_write", [] { throw std::runtime_error("crash after write"); });
  threw = false;
  try { (void)wal2.Append("t", *aster_test::MakeLineitem(10)); } catch (const std::runtime_error&) { threw = true; }
  fi.Reset();
  ASTER_CHECK(threw);
  WriteAheadLog wal3(cfg.wal_dir);
  ASTER_CHECK_OK(wal3.Open());
  int seen = 0;
  ASTER_CHECK_OK(wal3.Replay([&](const WriteAheadLog::Record&) { ++seen; return Status::OK(); }));
  ASTER_CHECK(seen >= 1);
}
