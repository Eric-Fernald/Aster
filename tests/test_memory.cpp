#include "aster/memory/bandwidth_probe.hpp"
#include "aster/memory/eviction.hpp"
#include "aster/memory/memory_manager.hpp"
#include "aster/memory/pool_allocator.hpp"
#include "aster/memory/spill_controller.hpp"
#include "aster/storage/segment.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::memory;

ASTER_TEST(bandwidth_table_defaults_and_multihop) {
  BandwidthTable t = BandwidthTable::Defaults(false);
  ASTER_CHECK(t.BytesPerSec(Tier::Host, Tier::Hbm) > 1e9);
  ASTER_CHECK(t.SecondsFor(Tier::Nvme, Tier::Hbm, 1 << 30) > 0);
  BandwidthTable coherent = BandwidthTable::Defaults(true);
  ASTER_CHECK(coherent.BytesPerSec(Tier::Host, Tier::Hbm) > t.BytesPerSec(Tier::Host, Tier::Hbm));
  std::string path = aster_test::TempDir("bw") + "/bw.tsv";
  ASTER_CHECK_OK(t.Save(path));
  BandwidthTable back;
  ASTER_CHECK_OK(back.Load(path));
  ASTER_CHECK_NEAR(back.BytesPerSec(Tier::Nvme, Tier::Host), t.BytesPerSec(Tier::Nvme, Tier::Host), 1.0);
}

ASTER_TEST(bandwidth_probe_runs_on_cpu) {
  ProbeOptions o;
  o.buffer_bytes = 1 << 20;
  o.iterations = 2;
  ASTER_ASSIGN_OK(BandwidthTable t, ProbeBandwidth(*hal::CpuDevice(), o));
  ASTER_CHECK(t.Has(Tier::Host, Tier::Hbm));
  ASTER_CHECK(t.BytesPerSec(Tier::Host, Tier::Hbm) > 0);
}

ASTER_TEST(pool_allocator_size_classes_and_arenas) {
  PoolAllocator pool(hal::CpuDevice(), 64 << 20, MemorySpace::HostPinned);
  ASTER_CHECK_EQ(PoolAllocator::SizeClass(100), size_t(256));
  ASTER_CHECK_EQ(PoolAllocator::SizeClass(300), size_t(512));
  ASTER_ASSIGN_OK(void* a, pool.Allocate(1000, 1));
  ASTER_ASSIGN_OK(void* b, pool.Allocate(1000, 1));
  ASTER_CHECK(a != b);
  ASTER_CHECK_EQ(pool.stats().in_use, size_t(2048));
  pool.ReleaseQuery(1);
  ASTER_CHECK_EQ(pool.stats().in_use, size_t(0));
  ASTER_ASSIGN_OK(void* c, pool.Allocate(1000, 2));
  ASTER_CHECK(c == a || c == b);
  ASTER_CHECK_EQ(pool.stats().cache_hits, uint64_t(1));
  auto too_big = pool.Allocate(size_t(128) << 20, 2);
  ASTER_CHECK(!too_big.ok());
}

ASTER_TEST(eviction_prefers_cold_unpinned_pages) {
  ResidencyMap map;
  for (uint32_t i = 0; i < 4; ++i) {
    PageDesc d;
    d.segment_id = 1; d.column_id = i; d.encoded_bytes = 100; d.tier = uint8_t(Tier::Hbm);
    d.device_ptr = reinterpret_cast<void*>(0x1000 + i);
    d.last_access_epoch = 10 + i;
    d.access_count = i == 0 ? 50 : 1;
    if (i == 2) d.pin_count = 1;
    if (i == 3) d.dimension_hint = true;
    map.Register(d);
  }
  EvictionPolicy pol;
  auto victims = pol.SelectVictims(map, Tier::Hbm, 150, 20);
  ASTER_CHECK_EQ(victims.size(), size_t(2));
  for (const auto& v : victims) { ASTER_CHECK(v.column_id != 2); ASTER_CHECK(v.column_id != 3); }
}

ASTER_TEST(memory_manager_tiers_and_eviction) {
  std::string dir = aster_test::TempDir("mm");
  auto batch = aster_test::MakeLineitem(20000);
  storage::SegmentWriteOptions o;
  ASTER_ASSIGN_OK(auto meta, storage::SegmentWriter::Write(*batch, 1, dir + "/s1.aseg", "t", o));
  EngineConfig cfg;
  cfg.mode = HardwareMode::CpuOnly;
  cfg.nvme_spill_dir = dir + "/spill";
  cfg.prefetch_depth = 2;
  size_t largest = 0;
  for (const auto& c : meta.columns) largest = std::max<size_t>(largest, c.bytes);
  cfg.vram_budget_bytes = largest * 3;  // room for about three pages
  MemoryManager mm(cfg, hal::CpuDevice(), BandwidthTable::Defaults(false));
  for (auto& d : meta.Pages()) mm.RegisterPage(d);
  ASTER_CHECK(mm.TierOf({1, 0}) == Tier::Nvme);
  {
    ASTER_ASSIGN_OK(PageHandle h, mm.Acquire({1, 0}));
    ASTER_CHECK(h.valid());
    ASTER_CHECK_EQ(h.bytes(), meta.columns[0].bytes);
    ASTER_CHECK(mm.TierOf({1, 0}) == Tier::Hbm);
    ASTER_CHECK(!mm.Evict({1, 0}).ok());  // pinned
  }
  ASTER_CHECK_OK(mm.Evict({1, 0}));
  ASTER_CHECK(mm.TierOf({1, 0}) == Tier::Host);
  MemoryStats s = mm.stats();
  ASTER_CHECK_EQ(s.bytes_nvme_to_host, uint64_t(meta.columns[0].bytes));
  ASTER_CHECK_EQ(s.evictions, uint64_t(1));

  // Pull more pages than fit; the manager must evict rather than fail.
  for (ColumnId c = 0; c < meta.columns.size(); ++c) {
    ASTER_ASSIGN_OK(PageHandle h, mm.Acquire({1, c}));
    ASTER_CHECK(h.valid());
  }
  ASTER_CHECK(mm.stats().evictions >= 2);
  ASTER_CHECK(mm.stats().hbm_in_use <= cfg.vram_budget_bytes);

  mm.Prefetch({{1, 5}, {1, 6}});
  ASTER_ASSIGN_OK(PageHandle h5, mm.Acquire({1, 5}));
  ASTER_CHECK(h5.valid());
  ASTER_CHECK(!mm.Acquire({99, 0}).ok());
}

ASTER_TEST(spill_controller_host_then_nvme) {
  std::string dir = aster_test::TempDir("spill");
  SpillController sc(hal::CpuDevice(), 1000, dir);
  std::vector<uint8_t> a(600, 1), b(600, 2);
  ASTER_ASSIGN_OK(SpillHandle ha, sc.SpillHost(a.data(), a.size(), SpillKind::HashTable, 0));
  ASTER_ASSIGN_OK(SpillHandle hb, sc.SpillHost(b.data(), b.size(), SpillKind::HashTable, 1));
  ASTER_CHECK(sc.host_bytes() <= 1000);
  ASTER_CHECK(sc.nvme_bytes() >= 600);
  std::vector<uint8_t> out(600);
  ASTER_CHECK_OK(sc.RestoreToHost(ha, out.data()));
  ASTER_CHECK_EQ(int(out[10]), 1);
  ASTER_CHECK_OK(sc.RestoreToHost(hb, out.data()));
  ASTER_CHECK_EQ(int(out[10]), 2);
  sc.Release(ha);
  sc.Release(hb);
  ASTER_CHECK_EQ(sc.host_bytes() + sc.nvme_bytes(), size_t(0));
}
