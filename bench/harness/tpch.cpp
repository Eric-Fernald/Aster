#include "tpch.hpp"

#include <cmath>
#include <random>

namespace aster::bench::tpch {

using namespace plan;

namespace {
int64_t DaysFromCivil(int y, int m, int d) {
  y -= m <= 2;
  int era = (y >= 0 ? y : y - 399) / 400;
  int yoe = y - era * 400;
  int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}
const char* kSegments[] = {"AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"};
const char* kPriorities[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
const char* kModes[] = {"REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"};
const char* kInstruct[] = {"DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"};
const char* kNations[] = {"ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN",
                          "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"};
const int kNationRegion[] = {0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
const char* kRegions[] = {"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"};
const char* kTypes1[] = {"STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"};
const char* kTypes2[] = {"ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"};
const char* kTypes3[] = {"TIN", "NICKEL", "BRASS", "STEEL", "COPPER"};
const char* kContainers[] = {"SM CASE", "SM BOX", "SM PACK", "SM PKG", "MED BAG", "MED BOX", "MED PKG", "MED PACK", "LG CASE", "LG BOX", "LG PACK", "LG PKG", "JUMBO CASE", "JUMBO BOX", "JUMBO PACK", "JUMBO PKG", "WRAP CASE", "WRAP BOX", "WRAP PACK", "WRAP PKG"};

struct Batches {
  const BatchSink& sink;
  int64_t batch_rows;
  Status Emit(const std::string& table, const Schema& schema, std::vector<Column> cols) {
    auto b = std::make_shared<RecordBatch>();
    b->schema = schema;
    b->columns = std::move(cols);
    return sink(table, b);
  }
};
}  // namespace

int32_t Date(int y, int m, int d) { return static_cast<int32_t>(DaysFromCivil(y, m, d)); }

Schema LineitemSchema() {
  return Schema{{{"l_orderkey", DataType::Of(TypeId::Int64), false}, {"l_partkey", DataType::Of(TypeId::Int64), false}, {"l_suppkey", DataType::Of(TypeId::Int64), false},
                 {"l_linenumber", DataType::Of(TypeId::Int32), false}, {"l_quantity", DataType::Of(TypeId::Int32), false}, {"l_extendedprice", DataType::Of(TypeId::Float64), false},
                 {"l_discount", DataType::Of(TypeId::Float64), false}, {"l_tax", DataType::Of(TypeId::Float64), false}, {"l_returnflag", DataType::Of(TypeId::String), false},
                 {"l_linestatus", DataType::Of(TypeId::String), false}, {"l_shipdate", DataType::Of(TypeId::Date32), false}, {"l_commitdate", DataType::Of(TypeId::Date32), false},
                 {"l_receiptdate", DataType::Of(TypeId::Date32), false}, {"l_shipinstruct", DataType::Of(TypeId::String), false}, {"l_shipmode", DataType::Of(TypeId::String), false},
                 {"l_comment", DataType::Of(TypeId::String), false}}};
}
Schema OrdersSchema() {
  return Schema{{{"o_orderkey", DataType::Of(TypeId::Int64), false}, {"o_custkey", DataType::Of(TypeId::Int64), false}, {"o_orderstatus", DataType::Of(TypeId::String), false},
                 {"o_totalprice", DataType::Of(TypeId::Float64), false}, {"o_orderdate", DataType::Of(TypeId::Date32), false}, {"o_orderpriority", DataType::Of(TypeId::String), false},
                 {"o_clerk", DataType::Of(TypeId::String), false}, {"o_shippriority", DataType::Of(TypeId::Int32), false}, {"o_comment", DataType::Of(TypeId::String), false}}};
}
Schema CustomerSchema() {
  return Schema{{{"c_custkey", DataType::Of(TypeId::Int64), false}, {"c_name", DataType::Of(TypeId::String), false}, {"c_address", DataType::Of(TypeId::String), false},
                 {"c_nationkey", DataType::Of(TypeId::Int64), false}, {"c_phone", DataType::Of(TypeId::String), false}, {"c_acctbal", DataType::Of(TypeId::Float64), false},
                 {"c_mktsegment", DataType::Of(TypeId::String), false}, {"c_comment", DataType::Of(TypeId::String), false}}};
}
Schema PartSchema() {
  return Schema{{{"p_partkey", DataType::Of(TypeId::Int64), false}, {"p_name", DataType::Of(TypeId::String), false}, {"p_mfgr", DataType::Of(TypeId::String), false},
                 {"p_brand", DataType::Of(TypeId::String), false}, {"p_type", DataType::Of(TypeId::String), false}, {"p_size", DataType::Of(TypeId::Int32), false},
                 {"p_container", DataType::Of(TypeId::String), false}, {"p_retailprice", DataType::Of(TypeId::Float64), false}, {"p_comment", DataType::Of(TypeId::String), false}}};
}
Schema SupplierSchema() {
  return Schema{{{"s_suppkey", DataType::Of(TypeId::Int64), false}, {"s_name", DataType::Of(TypeId::String), false}, {"s_address", DataType::Of(TypeId::String), false},
                 {"s_nationkey", DataType::Of(TypeId::Int64), false}, {"s_phone", DataType::Of(TypeId::String), false}, {"s_acctbal", DataType::Of(TypeId::Float64), false},
                 {"s_comment", DataType::Of(TypeId::String), false}}};
}
Schema PartsuppSchema() {
  return Schema{{{"ps_partkey", DataType::Of(TypeId::Int64), false}, {"ps_suppkey", DataType::Of(TypeId::Int64), false}, {"ps_availqty", DataType::Of(TypeId::Int32), false},
                 {"ps_supplycost", DataType::Of(TypeId::Float64), false}, {"ps_comment", DataType::Of(TypeId::String), false}}};
}
Schema NationSchema() {
  return Schema{{{"n_nationkey", DataType::Of(TypeId::Int64), false}, {"n_name", DataType::Of(TypeId::String), false}, {"n_regionkey", DataType::Of(TypeId::Int64), false}, {"n_comment", DataType::Of(TypeId::String), false}}};
}
Schema RegionSchema() {
  return Schema{{{"r_regionkey", DataType::Of(TypeId::Int64), false}, {"r_name", DataType::Of(TypeId::String), false}, {"r_comment", DataType::Of(TypeId::String), false}}};
}

Status Generate(const GenOptions& opts, const BatchSink& sink) {
  Batches out{sink, opts.batch_rows};
  std::mt19937_64 rng(opts.seed);
  const double sf = opts.scale_factor;
  const int64_t n_cust = std::max<int64_t>(1, static_cast<int64_t>(150000 * sf));
  const int64_t n_part = std::max<int64_t>(1, static_cast<int64_t>(200000 * sf));
  const int64_t n_supp = std::max<int64_t>(1, static_cast<int64_t>(10000 * sf));
  const int64_t n_orders = std::max<int64_t>(1, static_cast<int64_t>(1500000 * sf));
  const int32_t d0 = Date(1992, 1, 1), d1 = Date(1998, 8, 2);
  auto word = [&](int len) { std::string s; for (int i = 0; i < len; ++i) s += static_cast<char>('a' + rng() % 26); return s; };

  // region, nation
  {
    std::vector<int64_t> rk; std::vector<std::string> rn, rc;
    for (int i = 0; i < 5; ++i) { rk.push_back(i); rn.push_back(kRegions[i]); rc.push_back(word(20)); }
    ASTER_RETURN_NOT_OK(out.Emit("region", RegionSchema(), {MakeColumn<int64_t>(TypeId::Int64, rk), MakeStringColumn(rn), MakeStringColumn(rc)}));
    std::vector<int64_t> nk, nr; std::vector<std::string> nn, nc;
    for (int i = 0; i < 25; ++i) { nk.push_back(i); nn.push_back(kNations[i]); nr.push_back(kNationRegion[i]); nc.push_back(word(20)); }
    ASTER_RETURN_NOT_OK(out.Emit("nation", NationSchema(), {MakeColumn<int64_t>(TypeId::Int64, nk), MakeStringColumn(nn), MakeColumn<int64_t>(TypeId::Int64, nr), MakeStringColumn(nc)}));
  }
  // supplier
  for (int64_t start = 0; start < n_supp; start += opts.batch_rows) {
    int64_t n = std::min(opts.batch_rows, n_supp - start);
    std::vector<int64_t> k(n), nk(n); std::vector<std::string> name(n), addr(n), phone(n), comment(n); std::vector<double> bal(n);
    for (int64_t i = 0; i < n; ++i) {
      k[i] = start + i + 1; name[i] = "Supplier#" + std::to_string(k[i]); addr[i] = word(15); nk[i] = static_cast<int64_t>(rng() % 25);
      phone[i] = std::to_string(10 + nk[i]) + "-" + std::to_string(100 + rng() % 900); bal[i] = -999.99 + double(rng() % 1099999) / 100.0;
      comment[i] = (rng() % 100 < 2) ? "Customer Complaints " + word(10) : word(25);
    }
    ASTER_RETURN_NOT_OK(out.Emit("supplier", SupplierSchema(), {MakeColumn<int64_t>(TypeId::Int64, k), MakeStringColumn(name), MakeStringColumn(addr), MakeColumn<int64_t>(TypeId::Int64, nk), MakeStringColumn(phone), MakeColumn<double>(TypeId::Float64, bal), MakeStringColumn(comment)}));
  }
  // customer
  for (int64_t start = 0; start < n_cust; start += opts.batch_rows) {
    int64_t n = std::min(opts.batch_rows, n_cust - start);
    std::vector<int64_t> k(n), nk(n); std::vector<std::string> name(n), addr(n), phone(n), seg(n), comment(n); std::vector<double> bal(n);
    for (int64_t i = 0; i < n; ++i) {
      k[i] = start + i + 1; name[i] = "Customer#" + std::to_string(k[i]); addr[i] = word(15); nk[i] = static_cast<int64_t>(rng() % 25);
      phone[i] = std::to_string(10 + nk[i]) + "-" + std::to_string(100 + rng() % 900); bal[i] = -999.99 + double(rng() % 1099999) / 100.0;
      seg[i] = kSegments[rng() % 5]; comment[i] = word(30);
    }
    ASTER_RETURN_NOT_OK(out.Emit("customer", CustomerSchema(), {MakeColumn<int64_t>(TypeId::Int64, k), MakeStringColumn(name), MakeStringColumn(addr), MakeColumn<int64_t>(TypeId::Int64, nk), MakeStringColumn(phone), MakeColumn<double>(TypeId::Float64, bal), MakeStringColumn(seg), MakeStringColumn(comment)}));
  }
  // part + partsupp
  for (int64_t start = 0; start < n_part; start += opts.batch_rows) {
    int64_t n = std::min(opts.batch_rows, n_part - start);
    std::vector<int64_t> k(n); std::vector<std::string> name(n), mfgr(n), brand(n), type(n), cont(n), comment(n); std::vector<int32_t> size(n); std::vector<double> price(n);
    std::vector<int64_t> psp, pss; std::vector<int32_t> psq; std::vector<double> psc; std::vector<std::string> pscm;
    for (int64_t i = 0; i < n; ++i) {
      k[i] = start + i + 1; name[i] = word(6) + " " + word(6) + " " + word(6); int m = 1 + rng() % 5; mfgr[i] = "Manufacturer#" + std::to_string(m);
      brand[i] = "Brand#" + std::to_string(m) + std::to_string(1 + rng() % 5); type[i] = std::string(kTypes1[rng() % 6]) + " " + kTypes2[rng() % 5] + " " + kTypes3[rng() % 5];
      size[i] = 1 + static_cast<int32_t>(rng() % 50); cont[i] = kContainers[rng() % 20]; price[i] = 900.0 + double(k[i] % 1000) + double(rng() % 100) / 100.0; comment[i] = word(12);
      for (int s = 0; s < 4; ++s) { psp.push_back(k[i]); pss.push_back(1 + (k[i] + s * (n_supp / 4 + 1)) % n_supp); psq.push_back(1 + static_cast<int32_t>(rng() % 9999)); psc.push_back(1.0 + double(rng() % 100000) / 100.0); pscm.push_back(word(20)); }
    }
    ASTER_RETURN_NOT_OK(out.Emit("part", PartSchema(), {MakeColumn<int64_t>(TypeId::Int64, k), MakeStringColumn(name), MakeStringColumn(mfgr), MakeStringColumn(brand), MakeStringColumn(type), MakeColumn<int32_t>(TypeId::Int32, size), MakeStringColumn(cont), MakeColumn<double>(TypeId::Float64, price), MakeStringColumn(comment)}));
    ASTER_RETURN_NOT_OK(out.Emit("partsupp", PartsuppSchema(), {MakeColumn<int64_t>(TypeId::Int64, psp), MakeColumn<int64_t>(TypeId::Int64, pss), MakeColumn<int32_t>(TypeId::Int32, psq), MakeColumn<double>(TypeId::Float64, psc), MakeStringColumn(pscm)}));
  }
  // orders + lineitem
  int64_t li_batch_rows = opts.batch_rows;
  std::vector<int64_t> lo, lp, ls; std::vector<int32_t> ln, lq, lsd, lcd, lrd; std::vector<double> lep, ldc, ltx; std::vector<std::string> lrf, lls, lsi, lsm, lcm;
  auto flush_li = [&]() -> Status {
    if (lo.empty()) return Status::OK();
    Status s = out.Emit("lineitem", LineitemSchema(), {MakeColumn<int64_t>(TypeId::Int64, lo), MakeColumn<int64_t>(TypeId::Int64, lp), MakeColumn<int64_t>(TypeId::Int64, ls), MakeColumn<int32_t>(TypeId::Int32, ln),
                        MakeColumn<int32_t>(TypeId::Int32, lq), MakeColumn<double>(TypeId::Float64, lep), MakeColumn<double>(TypeId::Float64, ldc), MakeColumn<double>(TypeId::Float64, ltx), MakeStringColumn(lrf), MakeStringColumn(lls),
                        MakeColumn<int32_t>(TypeId::Date32, lsd), MakeColumn<int32_t>(TypeId::Date32, lcd), MakeColumn<int32_t>(TypeId::Date32, lrd), MakeStringColumn(lsi), MakeStringColumn(lsm), MakeStringColumn(lcm)});
    lo.clear(); lp.clear(); ls.clear(); ln.clear(); lq.clear(); lsd.clear(); lcd.clear(); lrd.clear(); lep.clear(); ldc.clear(); ltx.clear(); lrf.clear(); lls.clear(); lsi.clear(); lsm.clear(); lcm.clear();
    return s;
  };
  for (int64_t start = 0; start < n_orders; start += opts.batch_rows) {
    int64_t n = std::min(opts.batch_rows, n_orders - start);
    std::vector<int64_t> ok(n), ck(n); std::vector<std::string> st(n), pr(n), clerk(n), cm(n); std::vector<double> tp(n); std::vector<int32_t> od(n), sp(n, 0);
    for (int64_t i = 0; i < n; ++i) {
      int64_t key = start + i + 1;
      ok[i] = key; ck[i] = 1 + static_cast<int64_t>(rng() % n_cust); od[i] = d0 + static_cast<int32_t>(rng() % (d1 - d0 - 151));
      pr[i] = kPriorities[rng() % 5]; clerk[i] = "Clerk#" + std::to_string(1 + rng() % std::max<int64_t>(1, n_supp)); cm[i] = word(30);
      int lines = 1 + static_cast<int>(rng() % 7);
      double total = 0;
      bool all_f = true, all_o = true;
      for (int l = 0; l < lines; ++l) {
        int32_t qty = 1 + static_cast<int32_t>(rng() % 50);
        int64_t pk = 1 + static_cast<int64_t>(rng() % n_part);
        double price = (900.0 + double(pk % 1000) + double(rng() % 100) / 100.0) * qty;
        double disc = double(rng() % 11) / 100.0, tax = double(rng() % 9) / 100.0;
        int32_t ship = od[i] + 1 + static_cast<int32_t>(rng() % 121);
        int32_t commit = od[i] + 30 + static_cast<int32_t>(rng() % 61);
        int32_t receipt = ship + 1 + static_cast<int32_t>(rng() % 30);
        std::string rf = receipt <= Date(1995, 6, 17) ? (rng() % 2 ? "R" : "A") : "N";
        std::string lst = ship > Date(1995, 6, 17) ? "O" : "F";
        all_f &= lst == "F"; all_o &= lst == "O";
        lo.push_back(key); lp.push_back(pk); ls.push_back(1 + (pk + l * (n_supp / 4 + 1)) % n_supp); ln.push_back(l + 1); lq.push_back(qty);
        lep.push_back(price); ldc.push_back(disc); ltx.push_back(tax); lrf.push_back(rf); lls.push_back(lst); lsd.push_back(ship); lcd.push_back(commit); lrd.push_back(receipt);
        lsi.push_back(kInstruct[rng() % 4]); lsm.push_back(kModes[rng() % 7]); lcm.push_back(word(20));
        total += price * (1 - disc) * (1 + tax);
      }
      tp[i] = total;
      st[i] = all_f ? "F" : all_o ? "O" : "P";
      if (static_cast<int64_t>(lo.size()) >= li_batch_rows) ASTER_RETURN_NOT_OK(flush_li());
    }
    ASTER_RETURN_NOT_OK(out.Emit("orders", OrdersSchema(), {MakeColumn<int64_t>(TypeId::Int64, ok), MakeColumn<int64_t>(TypeId::Int64, ck), MakeStringColumn(st), MakeColumn<double>(TypeId::Float64, tp), MakeColumn<int32_t>(TypeId::Date32, od), MakeStringColumn(pr), MakeStringColumn(clerk), MakeColumn<int32_t>(TypeId::Int32, sp), MakeStringColumn(cm)}));
  }
  return flush_li();
}

Status LoadInto(Engine& engine, const GenOptions& opts) {
  struct T { const char* name; Schema schema; bool dim; };
  std::vector<T> tables = {{"lineitem", LineitemSchema(), false}, {"orders", OrdersSchema(), false}, {"customer", CustomerSchema(), true}, {"part", PartSchema(), true},
                           {"supplier", SupplierSchema(), true}, {"partsupp", PartsuppSchema(), false}, {"nation", NationSchema(), true}, {"region", RegionSchema(), true}};
  for (const auto& t : tables) {
    if (engine.catalog().HasTable(t.name)) ASTER_RETURN_NOT_OK(engine.DropTable(t.name));
    storage::TableInfo info;
    info.name = t.name; info.schema = t.schema; info.dimension_hint = t.dim;
    ASTER_RETURN_NOT_OK(engine.CreateTable(info));
  }
  std::map<std::string, std::vector<RecordBatchPtr>> pending;
  size_t pending_bytes = 0;
  auto flush = [&](const std::string& table) -> Status {
    auto& v = pending[table];
    if (v.empty()) return Status::OK();
    Status s = engine.LoadBatches(table, v);
    for (const auto& b : v) pending_bytes -= b->nbytes();
    v.clear();
    return s;
  };
  ASTER_RETURN_NOT_OK(Generate(opts, [&](const std::string& table, RecordBatchPtr b) -> Status {
    pending[table].push_back(b);
    pending_bytes += b->nbytes();
    if (pending_bytes > engine.config().segment_target_bytes * 4) return flush(table);
    return Status::OK();
  }));
  for (auto& [t, v] : pending) ASTER_RETURN_NOT_OK(flush(t));
  return Status::OK();
}

namespace {
Schema S(Engine& e, const char* t) { auto s = e.TableSchema(t); return s.ok() ? s.value() : Schema{}; }
int F(const Schema& s, const char* name) { return s.FieldIndex(name); }

RelPtr Q1(Engine& e) {
  Schema li = S(e, "lineitem");
  RelPtr read = PlanBuilder::Read("lineitem", li);
  RelPtr f = PlanBuilder::Filter(read, Call("lte", {Col(F(li, "l_shipdate")), Lit(int64_t(Date(1998, 9, 2)))}));
  int q = F(li, "l_quantity"), p = F(li, "l_extendedprice"), d = F(li, "l_discount"), t = F(li, "l_tax");
  std::vector<AggregateFn> aggs = {
      {"sum", {Col(q)}}, {"sum", {Col(p)}}, {"sum", {Call("multiply", {Col(p), Call("subtract", {Lit(1.0), Col(d)})})}},
      {"sum", {Call("multiply", {Call("multiply", {Col(p), Call("subtract", {Lit(1.0), Col(d)})}), Call("add", {Lit(1.0), Col(t)})})}},
      {"avg", {Col(q)}}, {"avg", {Col(p)}}, {"avg", {Col(d)}}, {"count_star", {}}};
  RelPtr agg = PlanBuilder::Aggregate(f, {Col(F(li, "l_returnflag")), Col(F(li, "l_linestatus"))}, aggs,
                                      {"l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"});
  return PlanBuilder::Sort(agg, {SortKey{Col(0), true}, SortKey{Col(1), true}});
}

RelPtr Q3(Engine& e) {
  Schema c = S(e, "customer"), o = S(e, "orders"), l = S(e, "lineitem");
  RelPtr cust = PlanBuilder::Filter(PlanBuilder::Read("customer", c), Call("equal", {Col(F(c, "c_mktsegment")), Lit(std::string("BUILDING"))}));
  RelPtr ord = PlanBuilder::Filter(PlanBuilder::Read("orders", o), Call("lt", {Col(F(o, "o_orderdate")), Lit(int64_t(Date(1995, 3, 15)))}));
  RelPtr li = PlanBuilder::Filter(PlanBuilder::Read("lineitem", l), Call("gt", {Col(F(l, "l_shipdate")), Lit(int64_t(Date(1995, 3, 15)))}));
  RelPtr oc = PlanBuilder::Join(ord, cust, JoinType::Inner, {F(o, "o_custkey")}, {F(c, "c_custkey")});
  RelPtr j = PlanBuilder::Join(li, oc, JoinType::Inner, {F(l, "l_orderkey")}, {F(o, "o_orderkey")});
  int nl = static_cast<int>(l.fields.size());
  RelPtr agg = PlanBuilder::Aggregate(j, {Col(F(l, "l_orderkey")), Col(nl + F(o, "o_orderdate")), Col(nl + F(o, "o_shippriority"))},
                                      {{"sum", {Call("multiply", {Col(F(l, "l_extendedprice")), Call("subtract", {Lit(1.0), Col(F(l, "l_discount"))})})}}},
                                      {"l_orderkey", "o_orderdate", "o_shippriority", "revenue"});
  RelPtr sorted = PlanBuilder::Sort(agg, {SortKey{Col(3), false}, SortKey{Col(1), true}});
  return PlanBuilder::Limit(sorted, 0, 10);
}

RelPtr Q5(Engine& e) {
  Schema c = S(e, "customer"), o = S(e, "orders"), l = S(e, "lineitem"), s = S(e, "supplier"), n = S(e, "nation"), r = S(e, "region");
  RelPtr reg = PlanBuilder::Filter(PlanBuilder::Read("region", r), Call("equal", {Col(F(r, "r_name")), Lit(std::string("ASIA"))}));
  RelPtr nat = PlanBuilder::Join(PlanBuilder::Read("nation", n), reg, JoinType::Inner, {F(n, "n_regionkey")}, {F(r, "r_regionkey")});
  RelPtr sup = PlanBuilder::Join(PlanBuilder::Read("supplier", s), nat, JoinType::Inner, {F(s, "s_nationkey")}, {F(n, "n_nationkey")});
  RelPtr ord = PlanBuilder::Filter(PlanBuilder::Read("orders", o), Call("and", {Call("gte", {Col(F(o, "o_orderdate")), Lit(int64_t(Date(1994, 1, 1)))}), Call("lt", {Col(F(o, "o_orderdate")), Lit(int64_t(Date(1995, 1, 1)))})}));
  RelPtr oc = PlanBuilder::Join(ord, PlanBuilder::Read("customer", c), JoinType::Inner, {F(o, "o_custkey")}, {F(c, "c_custkey")});
  RelPtr li = PlanBuilder::Read("lineitem", l);
  RelPtr ls = PlanBuilder::Join(li, sup, JoinType::Inner, {F(l, "l_suppkey")}, {F(s, "s_suppkey")});
  int nl = static_cast<int>(l.fields.size()), ns = static_cast<int>(s.fields.size()), nn = static_cast<int>(n.fields.size());
  RelPtr j = PlanBuilder::Join(ls, oc, JoinType::Inner, {F(l, "l_orderkey")}, {F(o, "o_orderkey")});
  RelPtr f = PlanBuilder::Filter(j, Call("equal", {Col(nl + F(s, "s_nationkey")), Col(nl + ns + nn + static_cast<int>(r.fields.size()) + static_cast<int>(o.fields.size()) + F(c, "c_nationkey"))}));
  RelPtr agg = PlanBuilder::Aggregate(f, {Col(nl + ns + F(n, "n_name"))}, {{"sum", {Call("multiply", {Col(F(l, "l_extendedprice")), Call("subtract", {Lit(1.0), Col(F(l, "l_discount"))})})}}}, {"n_name", "revenue"});
  return PlanBuilder::Sort(agg, {SortKey{Col(1), false}});
}

RelPtr Q6(Engine& e) {
  Schema li = S(e, "lineitem");
  RelPtr read = PlanBuilder::Read("lineitem", li);
  ExprPtr pred = Call("and", {Call("gte", {Col(F(li, "l_shipdate")), Lit(int64_t(Date(1994, 1, 1)))}), Call("lt", {Col(F(li, "l_shipdate")), Lit(int64_t(Date(1995, 1, 1)))}),
                              Call("between", {Col(F(li, "l_discount")), Lit(0.05), Lit(0.07)}), Call("lt", {Col(F(li, "l_quantity")), Lit(int64_t(24))})});
  RelPtr f = PlanBuilder::Filter(read, pred);
  return PlanBuilder::Aggregate(f, {}, {{"sum", {Call("multiply", {Col(F(li, "l_extendedprice")), Col(F(li, "l_discount"))})}}}, {"revenue"});
}

RelPtr Q10(Engine& e) {
  Schema c = S(e, "customer"), o = S(e, "orders"), l = S(e, "lineitem"), n = S(e, "nation");
  RelPtr ord = PlanBuilder::Filter(PlanBuilder::Read("orders", o), Call("and", {Call("gte", {Col(F(o, "o_orderdate")), Lit(int64_t(Date(1993, 10, 1)))}), Call("lt", {Col(F(o, "o_orderdate")), Lit(int64_t(Date(1994, 1, 1)))})}));
  RelPtr li = PlanBuilder::Filter(PlanBuilder::Read("lineitem", l), Call("equal", {Col(F(l, "l_returnflag")), Lit(std::string("R"))}));
  RelPtr cn = PlanBuilder::Join(PlanBuilder::Read("customer", c), PlanBuilder::Read("nation", n), JoinType::Inner, {F(c, "c_nationkey")}, {F(n, "n_nationkey")});
  RelPtr ocn = PlanBuilder::Join(ord, cn, JoinType::Inner, {F(o, "o_custkey")}, {F(c, "c_custkey")});
  RelPtr j = PlanBuilder::Join(li, ocn, JoinType::Inner, {F(l, "l_orderkey")}, {F(o, "o_orderkey")});
  int nl = static_cast<int>(l.fields.size()), no = static_cast<int>(o.fields.size()), nc = static_cast<int>(c.fields.size());
  RelPtr agg = PlanBuilder::Aggregate(j, {Col(nl + no + F(c, "c_custkey")), Col(nl + no + F(c, "c_name")), Col(nl + no + F(c, "c_acctbal")), Col(nl + no + nc + F(n, "n_name")), Col(nl + no + F(c, "c_address")), Col(nl + no + F(c, "c_phone")), Col(nl + no + F(c, "c_comment"))},
                                      {{"sum", {Call("multiply", {Col(F(l, "l_extendedprice")), Call("subtract", {Lit(1.0), Col(F(l, "l_discount"))})})}}},
                                      {"c_custkey", "c_name", "c_acctbal", "n_name", "c_address", "c_phone", "c_comment", "revenue"});
  return PlanBuilder::Limit(PlanBuilder::Sort(agg, {SortKey{Col(7), false}}), 0, 20);
}

RelPtr Q12(Engine& e) {
  Schema o = S(e, "orders"), l = S(e, "lineitem");
  RelPtr li = PlanBuilder::Filter(PlanBuilder::Read("lineitem", l), Call("and", {Call("in", {Col(F(l, "l_shipmode")), Lit(std::string("MAIL")), Lit(std::string("SHIP"))}),
                                  Call("lt", {Col(F(l, "l_commitdate")), Col(F(l, "l_receiptdate"))}), Call("lt", {Col(F(l, "l_shipdate")), Col(F(l, "l_commitdate"))}),
                                  Call("gte", {Col(F(l, "l_receiptdate")), Lit(int64_t(Date(1994, 1, 1)))}), Call("lt", {Col(F(l, "l_receiptdate")), Lit(int64_t(Date(1995, 1, 1)))})}));
  RelPtr j = PlanBuilder::Join(li, PlanBuilder::Read("orders", o), JoinType::Inner, {F(l, "l_orderkey")}, {F(o, "o_orderkey")});
  int nl = static_cast<int>(l.fields.size());
  ExprPtr high = Call("if_then", {Call("in", {Col(nl + F(o, "o_orderpriority")), Lit(std::string("1-URGENT")), Lit(std::string("2-HIGH"))}), Lit(int64_t(1)), Lit(int64_t(0))});
  ExprPtr low = Call("if_then", {Call("in", {Col(nl + F(o, "o_orderpriority")), Lit(std::string("1-URGENT")), Lit(std::string("2-HIGH"))}), Lit(int64_t(0)), Lit(int64_t(1))});
  RelPtr agg = PlanBuilder::Aggregate(j, {Col(F(l, "l_shipmode"))}, {{"sum", {high}}, {"sum", {low}}}, {"l_shipmode", "high_line_count", "low_line_count"});
  return PlanBuilder::Sort(agg, {SortKey{Col(0), true}});
}

RelPtr Q14(Engine& e) {
  Schema p = S(e, "part"), l = S(e, "lineitem");
  RelPtr li = PlanBuilder::Filter(PlanBuilder::Read("lineitem", l), Call("and", {Call("gte", {Col(F(l, "l_shipdate")), Lit(int64_t(Date(1995, 9, 1)))}), Call("lt", {Col(F(l, "l_shipdate")), Lit(int64_t(Date(1995, 10, 1)))})}));
  RelPtr j = PlanBuilder::Join(li, PlanBuilder::Read("part", p), JoinType::Inner, {F(l, "l_partkey")}, {F(p, "p_partkey")});
  int nl = static_cast<int>(l.fields.size());
  ExprPtr rev = Call("multiply", {Col(F(l, "l_extendedprice")), Call("subtract", {Lit(1.0), Col(F(l, "l_discount"))})});
  ExprPtr promo = Call("if_then", {Call("starts_with", {Col(nl + F(p, "p_type")), Lit(std::string("PROMO"))}), rev, Lit(0.0)});
  RelPtr agg = PlanBuilder::Aggregate(j, {}, {{"sum", {promo}}, {"sum", {rev}}}, {"promo", "total"});
  return PlanBuilder::Project(agg, {Call("divide", {Call("multiply", {Lit(100.0), Col(0)}), Col(1)})}, {"promo_revenue"});
}

RelPtr Q19(Engine& e) {
  Schema p = S(e, "part"), l = S(e, "lineitem");
  RelPtr li = PlanBuilder::Filter(PlanBuilder::Read("lineitem", l), Call("and", {Call("in", {Col(F(l, "l_shipmode")), Lit(std::string("AIR")), Lit(std::string("AIR REG"))}), Call("equal", {Col(F(l, "l_shipinstruct")), Lit(std::string("DELIVER IN PERSON"))})}));
  RelPtr j = PlanBuilder::Join(li, PlanBuilder::Read("part", p), JoinType::Inner, {F(l, "l_partkey")}, {F(p, "p_partkey")});
  int nl = static_cast<int>(l.fields.size());
  auto branch = [&](const char* brand, int64_t qlo, int64_t qhi, int64_t size_hi) {
    return Call("and", {Call("equal", {Col(nl + F(p, "p_brand")), Lit(std::string(brand))}), Call("between", {Col(F(l, "l_quantity")), Lit(qlo), Lit(qhi)}), Call("between", {Col(nl + F(p, "p_size")), Lit(int64_t(1)), Lit(size_hi)})});
  };
  RelPtr f = PlanBuilder::Filter(j, Call("or", {branch("Brand#12", 1, 11, 5), branch("Brand#23", 10, 20, 10), branch("Brand#34", 20, 30, 15)}));
  return PlanBuilder::Aggregate(f, {}, {{"sum", {Call("multiply", {Col(F(l, "l_extendedprice")), Call("subtract", {Lit(1.0), Col(F(l, "l_discount"))})})}}}, {"revenue"});
}
}  // namespace

std::vector<Query> Queries() {
  return {{"q1", Q1}, {"q3", Q3}, {"q5", Q5}, {"q6", Q6}, {"q10", Q10}, {"q12", Q12}, {"q14", Q14}, {"q19", Q19}};
}

}  // namespace aster::bench::tpch
