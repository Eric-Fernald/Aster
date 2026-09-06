#pragma once
#include <functional>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/engine.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::bench::tpch {

// dbgen-lite: statistically similar TPC-H tables at a scale factor, generated in batches.
struct GenOptions {
  double scale_factor = 0.01;
  int64_t batch_rows = 1 << 18;
  uint64_t seed = 20260905;
};

Schema LineitemSchema();
Schema OrdersSchema();
Schema CustomerSchema();
Schema PartSchema();
Schema SupplierSchema();
Schema PartsuppSchema();
Schema NationSchema();
Schema RegionSchema();

using BatchSink = std::function<Status(const std::string& table, RecordBatchPtr batch)>;
Status Generate(const GenOptions& opts, const BatchSink& sink);
Status LoadInto(Engine& engine, const GenOptions& opts);

// Queries expressed in the internal IR (what a Substrait producer would emit for the same SQL).
struct Query {
  std::string name;
  std::function<plan::RelPtr(Engine&)> build;
};
std::vector<Query> Queries();
int32_t Date(int y, int m, int d);

}  // namespace aster::bench::tpch
