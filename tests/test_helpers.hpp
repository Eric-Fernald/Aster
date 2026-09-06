#pragma once
#include <random>
#include <string>
#include <vector>

#include "aster/common/column.hpp"

namespace aster_test {

// A lineitem-like table: enough shape for filter, project, aggregate, join and pruning tests.
inline aster::Schema LineitemSchema() {
  using namespace aster;
  return Schema{{{"l_orderkey", DataType::Of(TypeId::Int64), false},
                 {"l_quantity", DataType::Of(TypeId::Int32), false},
                 {"l_extendedprice", DataType::Of(TypeId::Float64), false},
                 {"l_discount", DataType::Of(TypeId::Float64), false},
                 {"l_tax", DataType::Of(TypeId::Float64), false},
                 {"l_returnflag", DataType::Of(TypeId::String), false},
                 {"l_linestatus", DataType::Of(TypeId::String), false},
                 {"l_shipdate", DataType::Of(TypeId::Date32), false}}};
}

inline aster::RecordBatchPtr MakeLineitem(int64_t rows, uint64_t seed = 42, int64_t first_orderkey = 1) {
  using namespace aster;
  std::mt19937_64 rng(seed);
  std::vector<int64_t> ok(rows); std::vector<int32_t> qty(rows); std::vector<double> price(rows), disc(rows), tax(rows);
  std::vector<std::string> rf(rows), ls(rows); std::vector<int32_t> ship(rows);
  const char* flags[] = {"A", "N", "R"};
  const char* status[] = {"F", "O"};
  for (int64_t i = 0; i < rows; ++i) {
    ok[i] = first_orderkey + i / 4;
    qty[i] = 1 + static_cast<int32_t>(rng() % 50);
    price[i] = 900.0 + double(rng() % 100000) / 10.0;
    disc[i] = double(rng() % 11) / 100.0;
    tax[i] = double(rng() % 9) / 100.0;
    rf[i] = flags[rng() % 3];
    ls[i] = status[rng() % 2];
    ship[i] = 8766 + static_cast<int32_t>(rng() % 2557);  // 1994-01-01 .. 2000-12-31 in days since epoch
  }
  auto b = std::make_shared<RecordBatch>();
  b->schema = LineitemSchema();
  b->columns = {MakeColumn<int64_t>(TypeId::Int64, ok), MakeColumn<int32_t>(TypeId::Int32, qty), MakeColumn<double>(TypeId::Float64, price),
                MakeColumn<double>(TypeId::Float64, disc), MakeColumn<double>(TypeId::Float64, tax), MakeStringColumn(rf), MakeStringColumn(ls),
                MakeColumn<int32_t>(TypeId::Date32, ship)};
  return b;
}

inline aster::Schema OrdersSchema() {
  using namespace aster;
  return Schema{{{"o_orderkey", DataType::Of(TypeId::Int64), false},
                 {"o_custkey", DataType::Of(TypeId::Int64), false},
                 {"o_orderpriority", DataType::Of(TypeId::String), false}}};
}

inline aster::RecordBatchPtr MakeOrders(int64_t rows, uint64_t seed = 7) {
  using namespace aster;
  std::mt19937_64 rng(seed);
  std::vector<int64_t> ok(rows), ck(rows); std::vector<std::string> pr(rows);
  const char* prio[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
  for (int64_t i = 0; i < rows; ++i) { ok[i] = i + 1; ck[i] = 1 + static_cast<int64_t>(rng() % 100); pr[i] = prio[rng() % 5]; }
  auto b = std::make_shared<RecordBatch>();
  b->schema = OrdersSchema();
  b->columns = {MakeColumn<int64_t>(TypeId::Int64, ok), MakeColumn<int64_t>(TypeId::Int64, ck), MakeStringColumn(pr)};
  return b;
}

}  // namespace aster_test
