#pragma once
#include <functional>
#include <string>
#include <vector>

#include "aster/engine.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::bench::clickbench {

struct GenOptions {
  int64_t rows = 1'000'000;
  int64_t batch_rows = 1 << 18;
  uint64_t seed = 99;
};

// A hits-like table with the columns the ClickBench subset touches: heavy on string filters and skipping.
Schema HitsSchema();
Status Generate(const GenOptions& opts, const std::function<Status(RecordBatchPtr)>& sink);
Status LoadInto(Engine& engine, const GenOptions& opts);

struct Query {
  std::string name;
  std::function<plan::RelPtr(Engine&)> build;
};
std::vector<Query> Queries();

}  // namespace aster::bench::clickbench
