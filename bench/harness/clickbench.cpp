#include "clickbench.hpp"

#include <random>

namespace aster::bench::clickbench {

using namespace plan;

Schema HitsSchema() {
  return Schema{{{"WatchID", DataType::Of(TypeId::Int64), false}, {"CounterID", DataType::Of(TypeId::Int32), false}, {"ClientIP", DataType::Of(TypeId::Int32), false},
                 {"EventDate", DataType::Of(TypeId::Date32), false}, {"EventTime", DataType::Of(TypeId::Int64), false}, {"UserID", DataType::Of(TypeId::Int64), false},
                 {"RegionID", DataType::Of(TypeId::Int32), false}, {"ResolutionWidth", DataType::Of(TypeId::Int32), false}, {"AdvEngineID", DataType::Of(TypeId::Int32), false},
                 {"URL", DataType::Of(TypeId::String), false}, {"SearchPhrase", DataType::Of(TypeId::String), false}, {"Referer", DataType::Of(TypeId::String), false},
                 {"MobilePhoneModel", DataType::Of(TypeId::String), false}, {"IsRefresh", DataType::Of(TypeId::Int32), false}, {"DontCountHits", DataType::Of(TypeId::Int32), false}}};
}

Status Generate(const GenOptions& opts, const std::function<Status(RecordBatchPtr)>& sink) {
  std::mt19937_64 rng(opts.seed);
  const char* phrases[] = {"", "", "", "", "", "", "", "", "", "weather", "cars", "flights to rome", "gpu database", "news", "recipes", "python tutorial"};
  const char* hosts[] = {"http://example.com/", "http://news.site/", "http://shop.store/", "http://mail.ru/", "http://search.engine/", "http://blog.io/"};
  const char* models[] = {"", "", "", "", "iPhone", "Galaxy", "Pixel", "Nokia"};
  int32_t d0 = 15000;
  for (int64_t start = 0; start < opts.rows; start += opts.batch_rows) {
    int64_t n = std::min(opts.batch_rows, opts.rows - start);
    std::vector<int64_t> watch(n), etime(n), user(n); std::vector<int32_t> counter(n), ip(n), date(n), region(n), width(n), adv(n), refresh(n), dont(n);
    std::vector<std::string> url(n), phrase(n), referer(n), model(n);
    for (int64_t i = 0; i < n; ++i) {
      watch[i] = static_cast<int64_t>(rng() >> 1); counter[i] = 1 + static_cast<int32_t>(rng() % 5000); ip[i] = static_cast<int32_t>(rng());
      date[i] = d0 + static_cast<int32_t>(i * 7 / std::max<int64_t>(1, n)) + static_cast<int32_t>(start / opts.batch_rows) * 7;  // clustered by time
      etime[i] = int64_t(date[i]) * 86400 + static_cast<int64_t>(rng() % 86400); user[i] = static_cast<int64_t>(rng() % 200000);
      region[i] = static_cast<int32_t>(rng() % 300); width[i] = 320 + static_cast<int32_t>(rng() % 2000); adv[i] = rng() % 10 == 0 ? 1 + static_cast<int32_t>(rng() % 20) : 0;
      url[i] = std::string(hosts[rng() % 6]) + "page/" + std::to_string(rng() % 100000); phrase[i] = phrases[rng() % 16];
      referer[i] = rng() % 3 == 0 ? std::string(hosts[rng() % 6]) : ""; model[i] = models[rng() % 8];
      refresh[i] = rng() % 20 == 0; dont[i] = rng() % 50 == 0;
    }
    auto b = std::make_shared<RecordBatch>();
    b->schema = HitsSchema();
    b->columns = {MakeColumn<int64_t>(TypeId::Int64, watch), MakeColumn<int32_t>(TypeId::Int32, counter), MakeColumn<int32_t>(TypeId::Int32, ip), MakeColumn<int32_t>(TypeId::Date32, date),
                  MakeColumn<int64_t>(TypeId::Int64, etime), MakeColumn<int64_t>(TypeId::Int64, user), MakeColumn<int32_t>(TypeId::Int32, region), MakeColumn<int32_t>(TypeId::Int32, width),
                  MakeColumn<int32_t>(TypeId::Int32, adv), MakeStringColumn(url), MakeStringColumn(phrase), MakeStringColumn(referer), MakeStringColumn(model),
                  MakeColumn<int32_t>(TypeId::Int32, refresh), MakeColumn<int32_t>(TypeId::Int32, dont)};
    ASTER_RETURN_NOT_OK(sink(b));
  }
  return Status::OK();
}

Status LoadInto(Engine& engine, const GenOptions& opts) {
  if (engine.catalog().HasTable("hits")) ASTER_RETURN_NOT_OK(engine.DropTable("hits"));
  storage::TableInfo info;
  info.name = "hits";
  info.schema = HitsSchema();
  ASTER_RETURN_NOT_OK(engine.CreateTable(info));
  std::vector<RecordBatchPtr> pending;
  size_t bytes = 0;
  ASTER_RETURN_NOT_OK(Generate(opts, [&](RecordBatchPtr b) -> Status {
    pending.push_back(b);
    bytes += b->nbytes();
    if (bytes > engine.config().segment_target_bytes * 4) { Status s = engine.LoadBatches("hits", pending); pending.clear(); bytes = 0; return s; }
    return Status::OK();
  }));
  if (!pending.empty()) return engine.LoadBatches("hits", pending);
  return Status::OK();
}

namespace {
Schema H(Engine& e) { auto s = e.TableSchema("hits"); return s.ok() ? s.value() : HitsSchema(); }
int F(const Schema& s, const char* n) { return s.FieldIndex(n); }
RelPtr Read(Engine& e) { return PlanBuilder::Read("hits", H(e)); }
}  // namespace

std::vector<Query> Queries() {
  return {
      {"cb0_count", [](Engine& e) { return PlanBuilder::Aggregate(Read(e), {}, {{"count_star", {}}}); }},
      {"cb1_count_adv", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(PlanBuilder::Filter(Read(e), Call("not_equal", {Col(F(h, "AdvEngineID")), Lit(int64_t(0))})), {}, {{"count_star", {}}}); }},
      {"cb2_sum_avg_count", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(Read(e), {}, {{"sum", {Col(F(h, "AdvEngineID"))}}, {"count_star", {}}, {"avg", {Col(F(h, "ResolutionWidth"))}}}); }},
      {"cb3_avg_userid", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(Read(e), {}, {{"avg", {Col(F(h, "UserID"))}}}); }},
      {"cb4_count_distinct_user", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(Read(e), {}, {{"count_distinct", {Col(F(h, "UserID"))}}}); }},
      {"cb5_count_distinct_phrase", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(Read(e), {}, {{"count_distinct", {Col(F(h, "SearchPhrase"))}}}); }},
      {"cb6_min_max_date", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(Read(e), {}, {{"min", {Col(F(h, "EventDate"))}}, {"max", {Col(F(h, "EventDate"))}}}); }},
      {"cb7_group_adv", [](Engine& e) { Schema h = H(e); RelPtr a = PlanBuilder::Aggregate(PlanBuilder::Filter(Read(e), Call("not_equal", {Col(F(h, "AdvEngineID")), Lit(int64_t(0))})), {Col(F(h, "AdvEngineID"))}, {{"count_star", {}}}); return PlanBuilder::Sort(a, {SortKey{Col(1), false}}); }},
      {"cb8_region_users", [](Engine& e) { Schema h = H(e); RelPtr a = PlanBuilder::Aggregate(Read(e), {Col(F(h, "RegionID"))}, {{"count_distinct", {Col(F(h, "UserID"))}}}); return PlanBuilder::Limit(PlanBuilder::Sort(a, {SortKey{Col(1), false}}), 0, 10); }},
      {"cb12_phrase_top", [](Engine& e) { Schema h = H(e); RelPtr f = PlanBuilder::Filter(Read(e), Call("not_equal", {Col(F(h, "SearchPhrase")), Lit(std::string(""))})); RelPtr a = PlanBuilder::Aggregate(f, {Col(F(h, "SearchPhrase"))}, {{"count_star", {}}}); return PlanBuilder::Limit(PlanBuilder::Sort(a, {SortKey{Col(1), false}}), 0, 10); }},
      {"cb20_url_like", [](Engine& e) { Schema h = H(e); return PlanBuilder::Aggregate(PlanBuilder::Filter(Read(e), Call("like", {Col(F(h, "URL")), Lit(std::string("%news%"))})), {}, {{"count_star", {}}}); }},
      {"cb22_phrase_url_filter", [](Engine& e) { Schema h = H(e); RelPtr f = PlanBuilder::Filter(Read(e), Call("and", {Call("like", {Col(F(h, "URL")), Lit(std::string("%shop%"))}), Call("not_equal", {Col(F(h, "SearchPhrase")), Lit(std::string(""))})})); RelPtr a = PlanBuilder::Aggregate(f, {Col(F(h, "SearchPhrase"))}, {{"min", {Col(F(h, "URL"))}}, {"count_star", {}}}); return PlanBuilder::Limit(PlanBuilder::Sort(a, {SortKey{Col(2), false}}), 0, 10); }},
      {"cb30_date_range", [](Engine& e) { Schema h = H(e); RelPtr f = PlanBuilder::Filter(Read(e), Call("and", {Call("gte", {Col(F(h, "EventDate")), Lit(int64_t(15010))}), Call("lte", {Col(F(h, "EventDate")), Lit(int64_t(15012))}), Call("equal", {Col(F(h, "IsRefresh")), Lit(int64_t(0))})})); RelPtr a = PlanBuilder::Aggregate(f, {Col(F(h, "CounterID"))}, {{"count_star", {}}}); return PlanBuilder::Limit(PlanBuilder::Sort(a, {SortKey{Col(1), false}}), 0, 10); }},
      {"cb35_width_buckets", [](Engine& e) { Schema h = H(e); RelPtr p = PlanBuilder::Project(Read(e), {Call("divide", {Cast(Col(F(h, "ResolutionWidth")), DataType::Of(TypeId::Int64)), Lit(int64_t(100))}), Col(F(h, "UserID"))}, {"bucket", "u"}); RelPtr a = PlanBuilder::Aggregate(p, {Col(0)}, {{"count_star", {}}}); return PlanBuilder::Sort(a, {SortKey{Col(0), true}}); }},
  };
}

}  // namespace aster::bench::clickbench
