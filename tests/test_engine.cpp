#include "aster/engine.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::plan;

namespace {
EngineConfig TestConfig(const std::string& root) {
  EngineConfig cfg;
  cfg.mode = HardwareMode::CpuOnly;
  cfg.data_dir = root + "/data";
  cfg.wal_dir = root + "/wal";
  cfg.nvme_spill_dir = root + "/spill";
  cfg.kernel_cache_dir = root + "/kernels";
  cfg.bandwidth_cache_path = root + "/bw.tsv";
  cfg.segment_target_bytes = 1 << 18;
  cfg.tile_rows = 4096;
  cfg.log_plans = false;
  cfg.vram_budget_bytes = size_t(64) << 20;
  return cfg;
}

std::unique_ptr<Engine> OpenWithLineitem(const std::string& root, int64_t rows) {
  auto e = Engine::Open(TestConfig(root));
  if (!e.ok()) throw aster_test::Failure{e.status().ToString()};
  storage::TableInfo info;
  info.name = "lineitem";
  info.schema = aster_test::LineitemSchema();
  auto s = e.value()->CreateTable(info);
  if (!s.ok()) throw aster_test::Failure{s.ToString()};
  std::vector<RecordBatchPtr> batches;
  for (int64_t off = 0, i = 0; off < rows; off += 10000, ++i) batches.push_back(aster_test::MakeLineitem(std::min<int64_t>(10000, rows - off), 100 + i, off / 4 + 1));
  s = e.value()->LoadBatches("lineitem", batches);
  if (!s.ok()) throw aster_test::Failure{s.ToString()};
  return std::move(e.value());
}

double ReferenceQ6(const std::vector<RecordBatchPtr>& batches, int32_t d0, int32_t d1, int32_t qty) {
  double rev = 0;
  for (const auto& b : batches)
    for (int64_t i = 0; i < b->num_rows(); ++i) {
      int32_t ship = b->columns[7].Values<int32_t>()[i];
      double disc = b->columns[3].Values<double>()[i];
      if (ship >= d0 && ship < d1 && disc >= 0.05 && disc <= 0.07 && b->columns[1].Values<int32_t>()[i] < qty)
        rev += b->columns[2].Values<double>()[i] * disc;
    }
  return rev;
}
}  // namespace

ASTER_TEST(engine_q6_filter_aggregate_matches_reference) {
  std::string root = aster_test::TempDir("engine_q6");
  auto engine = OpenWithLineitem(root, 50000);
  ASTER_ASSIGN_OK(Schema schema, engine->TableSchema("lineitem"));
  RelPtr read = PlanBuilder::Read("lineitem", schema);
  ExprPtr pred = Call("and", {Call("gte", {Col(7), Lit(int64_t(9131))}), Call("lt", {Col(7), Lit(int64_t(9496))}),
                              Call("between", {Col(3), Lit(0.05), Lit(0.07)}), Call("lt", {Col(1), Lit(int64_t(24))})});
  RelPtr filter = PlanBuilder::Filter(read, pred);
  RelPtr agg = PlanBuilder::Aggregate(filter, {}, {AggregateFn{"sum", {Call("multiply", {Col(2), Col(3)})}}}, {"revenue"});
  ASTER_ASSIGN_OK(auto result, engine->Query(agg));
  ASTER_CHECK_EQ(result.num_rows(), 1);
  std::vector<RecordBatchPtr> ref;
  for (int64_t off = 0, i = 0; off < 50000; off += 10000, ++i) ref.push_back(aster_test::MakeLineitem(10000, 100 + i, off / 4 + 1));
  double expected = ReferenceQ6(ref, 9131, 9496, 24);
  ASTER_CHECK_NEAR(result.batches[0]->columns[0].Values<double>()[0], expected, 1e-6 * std::max(1.0, std::fabs(expected)));
  ASTER_CHECK(result.metrics.rows_scanned > 0);
  ASTER_CHECK(result.metrics.segments_scanned + result.metrics.segments_pruned > 0);
  ASTER_CHECK(result.metrics.cost_usd > 0);
  ASTER_CHECK_EQ(engine->history().size(), size_t(1));
}

ASTER_TEST(engine_q1_group_by_with_dictionary_keys) {
  std::string root = aster_test::TempDir("engine_q1");
  auto engine = OpenWithLineitem(root, 30000);
  ASTER_ASSIGN_OK(Schema schema, engine->TableSchema("lineitem"));
  RelPtr read = PlanBuilder::Read("lineitem", schema);
  RelPtr filter = PlanBuilder::Filter(read, Call("lte", {Col(7), Lit(int64_t(10000))}));
  std::vector<AggregateFn> aggs = {AggregateFn{"sum", {Col(1)}}, AggregateFn{"sum", {Col(2)}},
                                   AggregateFn{"sum", {Call("multiply", {Col(2), Call("subtract", {Lit(1.0), Col(3)})})}},
                                   AggregateFn{"avg", {Col(1)}}, AggregateFn{"count_star", {}}};
  RelPtr agg = PlanBuilder::Aggregate(filter, {Col(5), Col(6)}, aggs, {"l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "avg_qty", "count_order"});
  RelPtr sorted = PlanBuilder::Sort(agg, {SortKey{Col(0), true}, SortKey{Col(1), true}});
  ASTER_ASSIGN_OK(auto result, engine->Query(sorted));
  ASTER_CHECK_EQ(result.num_rows(), 6);
  auto out = result.Concat();
  int64_t total = 0;
  for (int64_t i = 0; i < out->num_rows(); ++i) total += out->columns[6].Values<int64_t>()[i];
  int64_t expected = 0;
  for (int64_t off = 0, i = 0; off < 30000; off += 10000, ++i) {
    auto b = aster_test::MakeLineitem(10000, 100 + i, off / 4 + 1);
    for (int64_t r = 0; r < b->num_rows(); ++r) if (b->columns[7].Values<int32_t>()[r] <= 10000) ++expected;
  }
  ASTER_CHECK_EQ(total, expected);
  ASTER_CHECK_EQ(std::string(out->columns[0].GetString(0)), "A");
  ASTER_CHECK_EQ(std::string(out->columns[1].GetString(0)), "F");
  ASTER_CHECK_EQ(std::string(out->columns[0].GetString(5)), "R");
}

ASTER_TEST(engine_join_orders_lineitem_top_k) {
  std::string root = aster_test::TempDir("engine_join");
  auto engine = OpenWithLineitem(root, 20000);
  storage::TableInfo info;
  info.name = "orders";
  info.schema = aster_test::OrdersSchema();
  info.dimension_hint = true;
  ASTER_CHECK_OK(engine->CreateTable(info));
  ASTER_CHECK_OK(engine->LoadBatches("orders", {aster_test::MakeOrders(3000)}));
  ASTER_ASSIGN_OK(Schema ls, engine->TableSchema("lineitem"));
  ASTER_ASSIGN_OK(Schema os, engine->TableSchema("orders"));
  RelPtr li = PlanBuilder::Read("lineitem", ls);
  RelPtr ord = PlanBuilder::Read("orders", os);
  RelPtr ordf = PlanBuilder::Filter(ord, Call("equal", {Col(2), Lit(std::string("1-URGENT"))}));
  RelPtr join = PlanBuilder::Join(li, ordf, JoinType::Inner, {0}, {0});
  RelPtr agg = PlanBuilder::Aggregate(join, {Col(9)}, {AggregateFn{"sum", {Col(2)}}, AggregateFn{"count_star", {}}}, {"custkey", "revenue", "n"});
  RelPtr sorted = PlanBuilder::Sort(agg, {SortKey{Col(1), false}});
  RelPtr top = PlanBuilder::Limit(sorted, 0, 5);
  ASTER_ASSIGN_OK(auto result, engine->Query(top));
  ASTER_CHECK_EQ(result.num_rows(), 5);
  auto out = result.Concat();
  for (int64_t i = 1; i < 5; ++i) ASTER_CHECK(out->columns[1].Values<double>()[i] <= out->columns[1].Values<double>()[i - 1]);
  ASTER_ASSIGN_OK(auto pp, engine->Explain(top));
  ASTER_CHECK(pp.pipelines.size() >= 3);
}

ASTER_TEST(engine_append_visible_before_and_after_compaction) {
  std::string root = aster_test::TempDir("engine_append");
  auto engine = OpenWithLineitem(root, 5000);
  ASTER_ASSIGN_OK(Schema schema, engine->TableSchema("lineitem"));
  auto count = [&]() -> int64_t {
    RelPtr read = PlanBuilder::Read("lineitem", schema);
    RelPtr agg = PlanBuilder::Aggregate(read, {}, {AggregateFn{"count_star", {}}});
    auto r = engine->Query(agg);
    if (!r.ok()) throw aster_test::Failure{r.status().ToString()};
    return r.value().batches[0]->columns[0].Values<int64_t>()[0];
  };
  ASTER_CHECK_EQ(count(), 5000);
  ASTER_CHECK_OK(engine->Append("lineitem", aster_test::MakeLineitem(1234)).status());
  ASTER_CHECK_EQ(count(), 6234);
  ASTER_CHECK_OK(engine->Compact("lineitem"));
  ASTER_CHECK_EQ(engine->delta().Rows("lineitem"), 0u);
  ASTER_CHECK_EQ(count(), 6234);
  ASTER_CHECK(engine->compactor().stats().segments_written >= 1);
}

ASTER_TEST(engine_recovers_after_reopen) {
  std::string root = aster_test::TempDir("engine_reopen");
  {
    auto engine = OpenWithLineitem(root, 3000);
    ASTER_CHECK_OK(engine->Append("lineitem", aster_test::MakeLineitem(500)).status());
  }
  ASTER_ASSIGN_OK(auto engine, Engine::Open(TestConfig(root)));
  ASTER_CHECK_EQ(engine->recovery_report().wal_records, 1u);
  ASTER_ASSIGN_OK(Schema schema, engine->TableSchema("lineitem"));
  RelPtr read = PlanBuilder::Read("lineitem", schema);
  RelPtr agg = PlanBuilder::Aggregate(read, {}, {AggregateFn{"count_star", {}}});
  ASTER_ASSIGN_OK(auto r, engine->Query(agg));
  ASTER_CHECK_EQ(r.batches[0]->columns[0].Values<int64_t>()[0], 3500);
}

ASTER_TEST(engine_external_input_and_cpu_fallback_function) {
  std::string root = aster_test::TempDir("engine_ext");
  ASTER_ASSIGN_OK(auto engine, Engine::Open(TestConfig(root)));
  auto b = aster_test::MakeOrders(100);
  RelPtr ext = PlanBuilder::External(b->schema, {b});
  RelPtr f = PlanBuilder::Filter(ext, Call("trim", {Col(2)}));  // trim is cpu only in the registry
  RelPtr proj = PlanBuilder::Project(f, {Col(0), Call("upper", {Col(2)})}, {"k", "p"});
  (void)f;
  RelPtr f2 = PlanBuilder::Filter(ext, Call("equal", {Call("trim", {Col(2)}), Lit(std::string("2-HIGH"))}));
  RelPtr proj2 = PlanBuilder::Project(f2, {Col(0), Call("upper", {Col(2)})}, {"k", "p"});
  ASTER_ASSIGN_OK(auto result, engine->Query(proj2));
  int64_t expected = 0;
  for (int64_t i = 0; i < 100; ++i) if (b->columns[2].GetString(i) == "2-HIGH") ++expected;
  ASTER_CHECK_EQ(result.num_rows(), expected);
  ASTER_CHECK(result.metrics.subtrees_cpu > 0);
}
