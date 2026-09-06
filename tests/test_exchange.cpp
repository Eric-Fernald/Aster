#include <thread>

#include "aster/exchange/exchange.hpp"
#include "aster/exchange/partitioner.hpp"
#include "aster/exchange/residency_directory.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::exchange;

ASTER_TEST(hash_partitioner_is_deterministic_and_complete) {
  auto b = aster_test::MakeLineitem(5000);
  HashPartitioner p({0}, 4);
  ASTER_ASSIGN_OK(auto parts, p.Split(*b));
  int64_t total = 0;
  for (const auto& part : parts) total += part->num_rows();
  ASTER_CHECK_EQ(total, 5000);
  ASTER_ASSIGN_OK(auto a1, p.Assign(*b));
  ASTER_ASSIGN_OK(auto a2, p.Assign(*b));
  ASTER_CHECK(a1 == a2);
  // Same key always lands in the same partition: rows sharing l_orderkey agree.
  for (int64_t i = 1; i < b->num_rows(); ++i)
    if (b->columns[0].Values<int64_t>()[i] == b->columns[0].Values<int64_t>()[i - 1]) ASTER_CHECK_EQ(a1[i], a1[i - 1]);
}

ASTER_TEST(in_process_shuffle_and_broadcast) {
  auto group = InProcessTransport::CreateGroup(3);
  std::vector<std::vector<RecordBatchPtr>> results(3), bcast(3);
  std::vector<std::thread> threads;
  for (int r = 0; r < 3; ++r) {
    threads.emplace_back([&, r] {
      ExchangeOperator ex(group[r], {0});
      auto local = aster_test::MakeLineitem(1000, r + 1, r * 1000 + 1);
      auto got = ex.Shuffle({local});
      if (got.ok()) results[r] = got.value();
      auto all = ex.Broadcast({local});
      if (all.ok()) bcast[r] = all.value();
    });
  }
  for (auto& t : threads) t.join();
  int64_t total = 0;
  for (auto& r : results) for (auto& b : r) total += b->num_rows();
  ASTER_CHECK_EQ(total, 3000);
  HashPartitioner p({0}, 3);
  for (int r = 0; r < 3; ++r)
    for (auto& b : results[r]) {
      ASTER_ASSIGN_OK(auto assign, p.Assign(*b));
      for (uint32_t a : assign) ASTER_CHECK_EQ(a, uint32_t(r));
    }
  for (int r = 0; r < 3; ++r) { int64_t n = 0; for (auto& b : bcast[r]) n += b->num_rows(); ASTER_CHECK_EQ(n, 3000); }
  ASTER_CHECK(ExchangeOperator::Choose(1000, 10, 25e9, false) == ExchangeStrategy::BroadcastSmaller);
  ASTER_CHECK(ExchangeOperator::Choose(1000, 900, 900e9, true) == ExchangeStrategy::HashShuffle);
}

ASTER_TEST(batch_serialization_roundtrip) {
  auto b = aster_test::MakeLineitem(77);
  auto bytes = SerializeBatch(*b);
  ASTER_ASSIGN_OK(auto back, DeserializeBatch(bytes.data(), bytes.size()));
  ASTER_CHECK_EQ(back->num_rows(), 77);
  ASTER_CHECK_EQ(std::string(back->columns[5].GetString(9)), std::string(b->columns[5].GetString(9)));
  ASTER_CHECK_NEAR(back->columns[2].Values<double>()[76], b->columns[2].Values<double>()[76], 1e-12);
}

ASTER_TEST(residency_directory_placement) {
  ResidencyDirectory dir;
  EngineConfig cfg;
  cfg.vram_budget_bytes = 1 << 20;
  memory::MemoryManager m0(cfg, hal::CpuDevice(), memory::BandwidthTable::Defaults(false));
  memory::MemoryManager m1(cfg, hal::CpuDevice(), memory::BandwidthTable::Defaults(false));
  dir.Attach(0, &m0);
  dir.Attach(1, &m1);
  memory::PageDesc d;
  d.segment_id = 1; d.column_id = 0; d.encoded_bytes = 100; d.tier = uint8_t(memory::Tier::Hbm);
  m1.RegisterPage(d);
  ASTER_CHECK_EQ(dir.BestGpuFor({{1, 0}}), 1);
  ASTER_CHECK_EQ(dir.HoldersOf({1, 0}).size(), size_t(1));
  dir.RegisterPartition("lineitem", 3, 0);
  ASTER_CHECK_EQ(dir.OwnerOf("lineitem", 3), 0);
  ASTER_CHECK_EQ(dir.OwnerOf("lineitem", 4), -1);
  ASTER_CHECK_EQ(dir.num_gpus(), 2);
}
