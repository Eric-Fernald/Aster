#include "aster/exec/backend.hpp"
#include "aster/exec/expression_eval.hpp"
#include "aster/exec/fused/codegen.hpp"
#include "aster/exec/fused/jit_compiler.hpp"
#include "aster/exec/fused/kernel_cache.hpp"
#include "aster/exec/operators/hash_aggregate.hpp"
#include "aster/exec/operators/hash_join.hpp"
#include "aster/exec/operators/sort.hpp"
#include "aster/exec/operators/vector_search.hpp"
#include "aster/exec/operators/window.hpp"
#include "aster/planner/pipeline_splitter.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::exec;
using namespace aster::plan;

ASTER_TEST(expression_arith_compare_and_strings) {
  auto b = std::make_shared<RecordBatch>();
  b->schema.fields = {{"a", DataType::Of(TypeId::Int64), true}, {"s", DataType::Of(TypeId::String), true}, {"d", DataType::Of(TypeId::Float64), true}};
  std::vector<bool> valid = {true, true, false};
  b->columns = {MakeColumn<int64_t>(TypeId::Int64, {1, 2, 3}, &valid), MakeStringColumn({"apple", "banana", "cherry"}), MakeColumn<double>(TypeId::Float64, {0.5, 1.5, 2.5})};
  ASTER_ASSIGN_OK(Column sum, ExpressionEvaluator::Evaluate(*Call("add", {Col(0), Lit(int64_t(10))}), *b));
  ASTER_CHECK_EQ(sum.Values<int64_t>()[1], 12);
  ASTER_CHECK(!sum.IsValid(2));
  ASTER_ASSIGN_OK(Column mul, ExpressionEvaluator::Evaluate(*Call("multiply", {Col(0), Col(2)}), *b));
  ASTER_CHECK_NEAR(mul.Values<double>()[1], 3.0, 1e-9);
  ASTER_ASSIGN_OK(Column like, ExpressionEvaluator::Evaluate(*Call("like", {Col(1), Lit(std::string("%an%"))}), *b));
  ASTER_CHECK_EQ(int(like.Values<uint8_t>()[0]), 0);
  ASTER_CHECK_EQ(int(like.Values<uint8_t>()[1]), 1);
  ASTER_ASSIGN_OK(SelectionVector sel, ExpressionEvaluator::Filter(*Call("and", {Call("gte", {Col(0), Lit(int64_t(2))}), Call("starts_with", {Col(1), Lit(std::string("b"))})}), *b, SelectionVector::All()));
  ASTER_CHECK_EQ(sel.rows.size(), size_t(1));
  ASTER_CHECK_EQ(sel.rows[0], 1u);
  ASTER_ASSIGN_OK(Column sub, ExpressionEvaluator::Evaluate(*Call("substring", {Col(1), Lit(int64_t(2)), Lit(int64_t(3))}), *b));
  ASTER_CHECK_EQ(std::string(sub.GetString(0)), "ppl");
  ASTER_ASSIGN_OK(Column cs, ExpressionEvaluator::Evaluate(*Call("if_then", {Call("gt", {Col(2), Lit(1.0)}), Lit(int64_t(1)), Lit(int64_t(0))}), *b));
  ASTER_CHECK_EQ(cs.Values<int64_t>()[0], 0);
  ASTER_CHECK_EQ(cs.Values<int64_t>()[2], 1);
  ASTER_ASSIGN_OK(Column casted, ExpressionEvaluator::Evaluate(*Cast(Col(2), DataType::Of(TypeId::Int32)), *b));
  ASTER_CHECK_EQ(casted.Values<int32_t>()[2], 2);
  ASTER_CHECK(ExpressionEvaluator::LikeMatch("hello world", "h%o w_rld"));
  ASTER_CHECK(!ExpressionEvaluator::LikeMatch("hello", "h_"));
}

ASTER_TEST(expression_dictionary_equality_on_codes) {
  auto b = std::make_shared<RecordBatch>();
  b->schema.fields = {{"r", DataType::Of(TypeId::String), false}};
  b->columns = {MakeDictionaryColumn({"EU", "US"}, {1, 0, 1, 1}, 5)};
  ASTER_ASSIGN_OK(SelectionVector sel, ExpressionEvaluator::Filter(*Call("equal", {Col(0), Lit(std::string("US"))}), *b, SelectionVector::All()));
  ASTER_CHECK_EQ(sel.rows.size(), size_t(3));
  ASTER_ASSIGN_OK(Column yr, ExpressionEvaluator::Evaluate(*Call("year", {Lit(int64_t(0))}), *b));
  (void)yr;
}

ASTER_TEST(date_extract_year) {
  auto b = std::make_shared<RecordBatch>();
  b->schema.fields = {{"d", DataType::Of(TypeId::Date32), false}};
  b->columns = {MakeColumn<int32_t>(TypeId::Date32, {0, 8766, 9131, 11323})};  // 1970-01-01, 1994-01-01, 1995-01-01, 2001-01-01
  ASTER_ASSIGN_OK(Column y, ExpressionEvaluator::Evaluate(*Call("year", {Col(0)}), *b));
  ASTER_CHECK_EQ(y.Values<int64_t>()[0], 1970);
  ASTER_CHECK_EQ(y.Values<int64_t>()[1], 1994);
  ASTER_CHECK_EQ(y.Values<int64_t>()[2], 1995);
  ASTER_CHECK_EQ(y.Values<int64_t>()[3], 2001);
}

ASTER_TEST(hash_aggregate_groups_and_merge) {
  auto b = aster_test::MakeLineitem(10000);
  std::vector<AggregateFn> aggs = {AggregateFn{"sum", {Col(1)}}, AggregateFn{"count_star", {}}, AggregateFn{"avg", {Col(2)}}, AggregateFn{"min", {Col(2)}}, AggregateFn{"max", {Col(5)}}, AggregateFn{"count_distinct", {Col(0)}}};
  RelPtr read = PlanBuilder::Read("l", b->schema);
  RelPtr agg = PlanBuilder::Aggregate(read, {Col(5), Col(6)}, aggs);
  HashAggregateState s1(agg->group_keys, agg->aggregates, b->schema, agg->output);
  HashAggregateState s2(agg->group_keys, agg->aggregates, b->schema, agg->output);
  auto half1 = SliceBatch(*b, 0, 5000), half2 = SliceBatch(*b, 5000, 5000);
  ASTER_CHECK_OK(s1.Update(*half1, SelectionVector::All(), nullptr));
  ASTER_CHECK_OK(s2.Update(*half2, SelectionVector::All(), nullptr));
  ASTER_CHECK_OK(s1.Merge(s2));
  ASTER_ASSIGN_OK(auto out, s1.Finalize());
  ASTER_CHECK_EQ(out->num_rows(), 6);
  int64_t total = 0;
  for (int64_t i = 0; i < out->num_rows(); ++i) total += out->columns[3].Values<int64_t>()[i];
  ASTER_CHECK_EQ(total, 10000);
  // Cross check one group against a direct computation.
  std::string key0(out->columns[0].GetString(0)), key1(out->columns[1].GetString(0));
  int64_t qsum = 0, cnt = 0;
  for (int64_t i = 0; i < b->num_rows(); ++i)
    if (b->columns[5].GetString(i) == key0 && b->columns[6].GetString(i) == key1) { qsum += b->columns[1].Values<int32_t>()[i]; ++cnt; }
  ASTER_CHECK_EQ(out->columns[2].Values<int64_t>()[0], qsum);
  ASTER_CHECK_EQ(out->columns[3].Values<int64_t>()[0], cnt);
  ASTER_CHECK(out->columns[7].Values<int64_t>()[0] > 0);
}

ASTER_TEST(hash_aggregate_over_runs) {
  auto b = std::make_shared<RecordBatch>();
  b->schema.fields = {{"k", DataType::Of(TypeId::Int32), false}, {"v", DataType::Of(TypeId::Int64), false}};
  b->columns = {MakeColumn<int32_t>(TypeId::Int32, {1, 2}), MakeColumn<int64_t>(TypeId::Int64, {10, 20})};
  std::vector<uint32_t> runs = {3, 2};
  RelPtr read = PlanBuilder::Read("t", b->schema);
  RelPtr agg = PlanBuilder::Aggregate(read, {Col(0)}, {AggregateFn{"sum", {Col(1)}}, AggregateFn{"count_star", {}}});
  HashAggregateState s(agg->group_keys, agg->aggregates, b->schema, agg->output);
  ASTER_CHECK_OK(s.Update(*b, SelectionVector::All(), &runs));
  ASTER_ASSIGN_OK(auto out, s.Finalize());
  ASTER_CHECK_EQ(out->num_rows(), 2);
  ASTER_CHECK_EQ(out->columns[1].Values<int64_t>()[0], 30);
  ASTER_CHECK_EQ(out->columns[2].Values<int64_t>()[1], 2);
}

ASTER_TEST(hash_join_inner_left_semi_anti) {
  auto li = aster_test::MakeLineitem(400);
  auto ord = aster_test::MakeOrders(50);
  CpuBackend backend;
  ExecContext ctx;
  Schema out_schema = li->schema;
  for (const auto& f : ord->schema.fields) out_schema.fields.push_back(f);
  ASTER_ASSIGN_OK(auto inner, backend.HashJoin(*ord, *li, JoinSpec{JoinType::Inner, {0}, {0}}, out_schema, ctx));
  int64_t expected = 0;
  for (int64_t i = 0; i < li->num_rows(); ++i) if (li->columns[0].Values<int64_t>()[i] <= 50) ++expected;
  ASTER_CHECK_EQ(inner->num_rows(), expected);
  for (int64_t i = 0; i < inner->num_rows(); ++i) ASTER_CHECK_EQ(inner->columns[0].Values<int64_t>()[i], inner->columns[8].Values<int64_t>()[i]);
  ASTER_ASSIGN_OK(auto left, backend.HashJoin(*ord, *li, JoinSpec{JoinType::Left, {0}, {0}}, out_schema, ctx));
  ASTER_CHECK_EQ(left->num_rows(), 400);
  ASTER_CHECK(left->columns[8].null_count > 0);
  ASTER_ASSIGN_OK(auto semi, backend.HashJoin(*ord, *li, JoinSpec{JoinType::Semi, {0}, {0}}, li->schema, ctx));
  ASTER_CHECK_EQ(semi->num_rows(), expected);
  ASTER_ASSIGN_OK(auto anti, backend.HashJoin(*ord, *li, JoinSpec{JoinType::Anti, {0}, {0}}, li->schema, ctx));
  ASTER_CHECK_EQ(anti->num_rows(), 400 - expected);
}

ASTER_TEST(hash_join_partitioned_build) {
  auto li = aster_test::MakeLineitem(2000);
  auto ord = aster_test::MakeOrders(500);
  HashJoinOptions opts;
  opts.build_budget_bytes = 1024;  // force radix partitioning
  Schema out_schema = li->schema;
  for (const auto& f : ord->schema.fields) out_schema.fields.push_back(f);
  HashJoinBuild build(JoinSpec{JoinType::Inner, {0}, {0}}, ord->schema, out_schema, opts);
  ASTER_CHECK_OK(build.Add(*ord));
  ExecContext ctx;
  ASTER_CHECK_OK(build.Finish(ctx));
  ASTER_CHECK(build.partitioned());
  ASTER_CHECK_EQ(build.num_partitions(), 16u);
  CpuBackend backend;
  ASTER_ASSIGN_OK(auto out, build.Probe(*li, backend, ctx));
  ASTER_CHECK_EQ(out->num_rows(), 2000);
}

ASTER_TEST(sort_topk_and_merge_runs) {
  auto b = aster_test::MakeLineitem(3000);
  CpuBackend backend;
  ExecContext ctx;
  SortSpec spec{{SortKey{Col(2), false}, SortKey{Col(0), true}}, 5};
  ASTER_ASSIGN_OK(auto top, backend.Sort(*b, spec, ctx));
  ASTER_CHECK_EQ(top->num_rows(), 5);
  for (int64_t i = 1; i < 5; ++i) ASTER_CHECK(top->columns[2].Values<double>()[i] <= top->columns[2].Values<double>()[i - 1]);
  SortAccumulator acc(SortSpec{{SortKey{Col(0), true}}, -1}, b->schema, 1);  // one run per batch
  for (int64_t off = 0; off < 3000; off += 1000) ASTER_CHECK_OK(acc.Add(SliceBatch(*b, off, 1000), backend, ctx));
  ASTER_CHECK_EQ(acc.num_runs(), size_t(3));
  ASTER_ASSIGN_OK(auto merged, acc.Finalize(backend, ctx));
  ASTER_CHECK_EQ(merged->num_rows(), 3000);
  for (int64_t i = 1; i < 3000; ++i) ASTER_CHECK(merged->columns[0].Values<int64_t>()[i] >= merged->columns[0].Values<int64_t>()[i - 1]);
}

ASTER_TEST(window_functions) {
  auto b = std::make_shared<RecordBatch>();
  b->schema.fields = {{"p", DataType::Of(TypeId::Int64), false}, {"v", DataType::Of(TypeId::Int64), false}};
  b->columns = {MakeColumn<int64_t>(TypeId::Int64, {1, 1, 2, 2, 2}), MakeColumn<int64_t>(TypeId::Int64, {5, 3, 9, 9, 1})};
  WindowFn rn{"row_number", {}, {Col(0)}, {SortKey{Col(1), true}}};
  WindowFn rk{"rank", {}, {Col(0)}, {SortKey{Col(1), true}}};
  WindowFn sm{"sum", {Col(1)}, {Col(0)}, {}};
  RelPtr read = PlanBuilder::Read("t", b->schema);
  RelPtr w = PlanBuilder::Window(read, {rn, rk, sm});
  WindowOperator op(w->windows, b->schema, w->output);
  ASTER_ASSIGN_OK(auto out, op.Execute(*b));
  ASTER_CHECK_EQ(out->columns.size(), size_t(5));
  ASTER_CHECK_EQ(out->columns[2].Values<int64_t>()[1], 1);  // v=3 first in partition 1
  ASTER_CHECK_EQ(out->columns[2].Values<int64_t>()[0], 2);
  ASTER_CHECK_EQ(out->columns[3].Values<int64_t>()[2], out->columns[3].Values<int64_t>()[3]);  // tie ranks
  ASTER_CHECK_EQ(out->columns[4].Values<int64_t>()[4], 19);
}

ASTER_TEST(vector_brute_force_search) {
  std::vector<float> data = {0, 0, 1, 1, 5, 5, 0.1f, 0.1f};
  Column vecs;
  vecs.type = DataType::Vector(2);
  vecs.length = 4;
  vecs.values = Buffer::CopyOf(data.data(), data.size() * sizeof(float));
  BruteForceIndex idx;
  ExecContext ctx;
  ASTER_CHECK_OK(idx.Build(vecs, ctx));
  ASTER_ASSIGN_OK(AnnResult r, idx.Search(AnnQuery{{0, 0}, 2, VectorMetric::L2}, ctx));
  ASTER_CHECK_EQ(r.row_ids.size(), size_t(2));
  ASTER_CHECK_EQ(r.row_ids[0], 0);
  ASTER_CHECK_EQ(r.row_ids[1], 3);
}

ASTER_TEST(codegen_produces_fused_kernel_source) {
  RelPtr read = PlanBuilder::Read("lineitem", aster_test::LineitemSchema());
  ExprPtr pred = Call("and", {Call("gte", {Col(7), Lit(int64_t(9000))}), Call("lt", {Col(1), Lit(int64_t(24))})});
  RelPtr filter = PlanBuilder::Filter(read, pred);
  RelPtr proj = PlanBuilder::Project(filter, {Call("multiply", {Col(2), Col(3)})}, {"rev"});
  RelPtr agg = PlanBuilder::Aggregate(proj, {}, {AggregateFn{"sum", {Col(0)}}});
  AssignNodeIds(agg);
  auto pipes = planner::PipelineSplitter::Split(agg);
  std::vector<fused::ColumnBinding> in;
  for (int i = 0; i < 8; ++i) in.push_back({i, aster_test::LineitemSchema().fields[i].type, i == 1 ? storage::Encoding::ForBitPack : storage::Encoding::Plain});
  ASTER_ASSIGN_OK(fused::KernelSpec spec, fused::Codegen::Generate(pipes[0], in, 32768));
  ASTER_CHECK(spec.source.find("__global__") != std::string::npos);
  ASTER_CHECK(spec.source.find("continue;") != std::string::npos);
  ASTER_CHECK(spec.source.find("atomicAdd") != std::string::npos);
  ASTER_CHECK(spec.has_partial_aggregate);
  ASTER_CHECK(spec.source.find("unpack(") != std::string::npos);
  fused::KernelCache cache(aster_test::TempDir("kcache"));
  fused::JitCompiler jit(cache);
  ASTER_ASSIGN_OK(auto k1, jit.GetOrCompile(spec));
  ASTER_CHECK(!k1->image.empty());
  ASTER_ASSIGN_OK(auto k2, jit.GetOrCompile(spec));
  ASTER_CHECK_EQ(cache.hits(), uint64_t(1));
  ASTER_CHECK(cache.Contains(spec.shape_hash));
  auto bad = fused::Codegen::ExprToCuda(*Call("regexp_match", {Col(0), Lit(std::string("x"))}), in, "row");
  ASTER_CHECK(!bad.ok());
}
