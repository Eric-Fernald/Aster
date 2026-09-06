#include "aster/integration/capability_registry.hpp"
#include "aster/integration/fallback_router.hpp"
#include "aster/planner/cost_model.hpp"
#include "aster/planner/pipeline_splitter.hpp"
#include "aster/planner/planner.hpp"
#include "aster/planner/predicate_rewrite.hpp"
#include "aster/planner/segment_pruner.hpp"
#include "aster/storage/segment.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::plan;
using namespace aster::planner;

namespace {
RelPtr Q6Like() {
  // SELECT sum(l_extendedprice * l_discount) FROM lineitem WHERE l_shipdate >= d1 AND l_quantity < 24
  RelPtr read = PlanBuilder::Read("lineitem", aster_test::LineitemSchema());
  ExprPtr pred = Call("and", {Call("gte", {Col(7), Lit(int64_t(9000))}), Call("lt", {Col(1), Lit(int64_t(24))})});
  RelPtr filter = PlanBuilder::Filter(read, pred);
  AggregateFn sum{"sum", {Call("multiply", {Col(2), Col(3)})}};
  return PlanBuilder::Aggregate(filter, {}, {sum}, {"revenue"});
}
}  // namespace

ASTER_TEST(capability_registry_defaults_and_tsv) {
  integration::CapabilityRegistry reg;
  ASTER_CHECK(reg.Function("equal") == integration::Support::Gpu);
  ASTER_CHECK(reg.Function("regexp_match") == integration::Support::Cpu);
  ASTER_CHECK(reg.Function("made_up") == integration::Support::Unsupported);
  ASTER_CHECK(reg.Relation(RelKind::Window) == integration::Support::Cpu);
  ASTER_CHECK(!reg.Gaps().empty());
  std::string path = aster_test::TempDir("caps") + "/caps.tsv";
  ASTER_CHECK_OK(reg.SaveTsv(path));
  integration::CapabilityRegistry loaded;
  loaded.Set("fn:equal", integration::Support::Cpu);
  ASTER_CHECK_OK(loaded.LoadTsv(path));
  ASTER_CHECK(loaded.Function("equal") == integration::Support::Gpu);
  std::string why;
  RelPtr read = PlanBuilder::Read("t", aster_test::LineitemSchema());
  RelPtr f = PlanBuilder::Filter(read, Call("regexp_match", {Col(5), Lit(std::string("A.*"))}));
  ASTER_CHECK(reg.CheckNode(*f, &why) == integration::Support::Cpu);
  ASTER_CHECK(why.find("regexp_match") != std::string::npos);
}

ASTER_TEST(fallback_router_marks_subtrees) {
  integration::CapabilityRegistry reg;
  memory::BandwidthTable bw = memory::BandwidthTable::Defaults(false);
  integration::FallbackRouter router(reg, bw);
  RelPtr read = PlanBuilder::Read("t", aster_test::LineitemSchema());
  RelPtr f = PlanBuilder::Filter(read, Call("trim", {Col(5)}));
  RelPtr agg = PlanBuilder::Aggregate(f, {Col(5)}, {AggregateFn{"count_star", {}}});
  AssignNodeIds(agg);
  auto stats = router.Route(agg, [](const Rel&) { return uint64_t(1000000); });
  ASTER_CHECK(f->placement == Placement::Cpu);
  ASTER_CHECK(agg->placement == Placement::Gpu);
  ASTER_CHECK_EQ(stats.nodes_total, 3u);
  ASTER_CHECK(stats.fallback_rate() > 0);
}

ASTER_TEST(pipeline_splitter_breaks_at_aggregate_and_join) {
  RelPtr q6 = Q6Like();
  AssignNodeIds(q6);
  auto pipes = PipelineSplitter::Split(q6);
  ASTER_CHECK_EQ(pipes.size(), size_t(1));
  ASTER_CHECK_EQ(pipes[0].ops.size(), size_t(3));
  ASTER_CHECK(pipes[0].fusable);
  ASTER_CHECK(!pipes[0].shape_hash.empty());

  RelPtr li = PlanBuilder::Read("lineitem", aster_test::LineitemSchema());
  RelPtr ord = PlanBuilder::Read("orders", aster_test::OrdersSchema());
  RelPtr join = PlanBuilder::Join(li, ord, JoinType::Inner, {0}, {0});
  RelPtr agg = PlanBuilder::Aggregate(join, {Col(10)}, {AggregateFn{"sum", {Col(2)}}});
  RelPtr sorted = PlanBuilder::Sort(agg, {SortKey{Col(1), false}});
  RelPtr top = PlanBuilder::Limit(sorted, 0, 10);
  AssignNodeIds(top);
  auto p2 = PipelineSplitter::Split(top);
  ASTER_CHECK_EQ(p2.size(), size_t(4));  // build(orders), probe+agg, sort, limit
  ASTER_CHECK(p2.back().ops.back()->kind == RelKind::Limit);
  ASTER_CHECK(!p2[1].depends_on.empty());
  std::string h1 = PipelineSplitter::ShapeHash(pipes[0]);
  RelPtr q6b = Q6Like();
  AssignNodeIds(q6b);
  ASTER_CHECK_EQ(h1, PipelineSplitter::ShapeHash(PipelineSplitter::Split(q6b)[0]));
}

ASTER_TEST(segment_pruner_uses_zone_maps_and_bloom) {
  std::string dir = aster_test::TempDir("prune");
  auto b1 = aster_test::MakeLineitem(2000, 1, 1);
  auto b2 = aster_test::MakeLineitem(2000, 2, 100000);
  ASTER_ASSIGN_OK(auto m1, storage::SegmentWriter::Write(*b1, 1, dir + "/1.aseg", "lineitem", {}));
  ASTER_ASSIGN_OK(auto m2, storage::SegmentWriter::Write(*b2, 2, dir + "/2.aseg", "lineitem", {}));
  std::vector<std::shared_ptr<storage::SegmentMeta>> segs = {std::make_shared<storage::SegmentMeta>(m1), std::make_shared<storage::SegmentMeta>(m2)};
  Schema schema = aster_test::LineitemSchema();
  std::vector<int> proj = {0, 1, 2, 3, 4, 5, 6, 7};
  ExprPtr p = Call("gt", {Col(0), Lit(int64_t(50000))});
  PruneResult r = SegmentPruner::Prune(segs, p.get(), schema, proj);
  ASTER_CHECK_EQ(r.kept.size(), size_t(1));
  ASTER_CHECK_EQ(r.kept[0]->segment_id, 2u);
  ASTER_CHECK_EQ(r.pruned_rows, 2000u);
  ExprPtr s = Call("equal", {Col(5), Lit(std::string("Z"))});
  ASTER_CHECK_EQ(SegmentPruner::Prune(segs, s.get(), schema, proj).kept.size(), size_t(0));
  ExprPtr in = Call("in", {Col(5), Lit(std::string("Z")), Lit(std::string("A"))});
  ASTER_CHECK_EQ(SegmentPruner::Prune(segs, in.get(), schema, proj).kept.size(), size_t(2));
  ExprPtr orp = Call("or", {p, Call("lt", {Col(0), Lit(int64_t(10))})});
  ASTER_CHECK_EQ(SegmentPruner::Prune(segs, orp.get(), schema, proj).kept.size(), size_t(2));
  RelPtr read = PlanBuilder::Read("lineitem", schema);
  RelPtr f = PlanBuilder::Filter(read, p);
  ExprPtr collected = SegmentPruner::CollectPredicates(read, f);
  ASTER_CHECK(collected != nullptr);
}

ASTER_TEST(predicate_rewrite_into_encoded_domain) {
  Column region = MakeStringColumn({"EU", "US", "EU", "ASIA"});
  storage::EncodeOptions o; o.encoding = storage::Encoding::Dictionary;
  ASTER_ASSIGN_OK(auto enc, storage::EncodeColumn(region, o));
  ASTER_ASSIGN_OK(auto dict, storage::ViewDictionary(enc.data.data(), enc.data.size()));
  Column qty = MakeColumn<int64_t>(TypeId::Int64, {10, 12, 15, 11});
  storage::EncodeOptions f; f.encoding = storage::Encoding::ForBitPack;
  ASTER_ASSIGN_OK(auto fenc, storage::EncodeColumn(qty, f));
  ASTER_ASSIGN_OK(auto frame, storage::ViewFor(fenc.data.data(), fenc.data.size()));
  std::map<int, EncodedDomain> domains;
  domains[0] = EncodedDomain{storage::Encoding::Dictionary, &dict, nullptr};
  domains[1] = EncodedDomain{storage::Encoding::ForBitPack, nullptr, &frame};

  ExprPtr eq = Call("equal", {Col(0), Lit(std::string("EU"))});
  ASTER_CHECK(PredicateRewriter::IsRewritable(*eq));
  ExprPtr r = PredicateRewriter::Rewrite(*eq, domains);
  ASTER_CHECK(r->encoded_domain);
  ASTER_CHECK_EQ(std::get<int64_t>(r->args[1]->literal), int64_t(0));
  ExprPtr missing = PredicateRewriter::Rewrite(*Call("equal", {Col(0), Lit(std::string("MARS"))}), domains);
  ASTER_CHECK_EQ(*PredicateRewriter::ConstantValue(*missing), false);
  ExprPtr lt = PredicateRewriter::Rewrite(*Call("lt", {Col(1), Lit(int64_t(12))}), domains);
  ASTER_CHECK(lt->encoded_domain);
  ASTER_CHECK_EQ(std::get<int64_t>(lt->args[1]->literal), int64_t(2));
  ExprPtr always = PredicateRewriter::Rewrite(*Call("gt", {Col(1), Lit(int64_t(5))}), domains);
  ASTER_CHECK_EQ(*PredicateRewriter::ConstantValue(*always), true);
  ExprPtr conj = PredicateRewriter::Rewrite(*Call("and", {eq, Call("gt", {Col(1), Lit(int64_t(5))})}), domains);
  ASTER_CHECK(conj->is_call("equal"));
}

ASTER_TEST(cost_model_estimates_rows_and_bytes) {
  memory::BandwidthTable bw = memory::BandwidthTable::Defaults(false);
  CostModel cost(bw, nullptr);
  RelPtr q6 = Q6Like();
  AssignNodeIds(q6);
  uint64_t rows = cost.EstimateRows(*q6->inputs[0], [](const Rel&) { return uint64_t(1000000); });
  ASTER_CHECK(rows < 1000000 && rows > 0);
  ASTER_CHECK_EQ(cost.EstimateRows(*q6, [](const Rel&) { return uint64_t(1000000); }), uint64_t(1));
  ASTER_CHECK_NEAR(cost.Selectivity(*Call("equal", {Col(0), Lit(int64_t(1))})), 0.1, 1e-9);
}

ASTER_TEST(planner_end_to_end_without_catalog) {
  EngineConfig cfg;
  cfg.mode = HardwareMode::Discrete;
  cfg.log_plans = false;
  PlannerContext ctx;
  ctx.config = &cfg;
  Planner planner(ctx);
  ASTER_ASSIGN_OK(PhysicalPlan pp, planner.Plan(Q6Like()));
  ASTER_CHECK_EQ(pp.pipelines.size(), size_t(1));
  ASTER_CHECK(pp.pipelines[0].placement == PlacementMode::Resident);
  ASTER_CHECK_EQ(pp.kernel_cache_misses, 1u);
  ASTER_CHECK(!pp.ToString().empty());
}
