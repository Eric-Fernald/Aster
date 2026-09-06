#pragma once
#include <functional>
#include <string>
#include <vector>

#include "aster/engine.hpp"
#include "aster/metrics/metrics.hpp"

namespace aster::bench {

struct BenchQuery {
  std::string name;
  std::function<plan::RelPtr(Engine&)> build;
};

struct RunResult {
  std::string query;
  int iterations = 0;
  double min_ms = 0, median_ms = 0, max_ms = 0;
  double cost_usd = 0;
  uint64_t bytes_moved = 0;
  uint64_t bytes_estimated = 0;
  uint64_t rows_scanned = 0;
  uint64_t rows_out = 0;
  double fallback_rate = 0;
  double kernel_cache_hit = 0;
  double hbm_util = 0;
  uint64_t segments_pruned = 0, segments_scanned = 0;
  std::string error;
  double estimate_error() const { return bytes_moved ? double(int64_t(bytes_estimated) - int64_t(bytes_moved)) / double(bytes_moved) : 0.0; }
};

struct HarnessOptions {
  int iterations = 3;
  int warmup = 1;
  double hourly_cost_usd = 3.0;
  std::string results_json;
  std::string results_csv;
  bool verbose = true;
};

// Runs queries, measures every metric in the design plan's table, and writes machine readable results.
class Harness {
 public:
  Harness(Engine& engine, HarnessOptions opts) : engine_(engine), opts_(std::move(opts)) {}
  RunResult Run(const BenchQuery& q);
  std::vector<RunResult> RunAll(const std::vector<BenchQuery>& queries);
  static std::string ToJson(const std::vector<RunResult>& results);
  static std::string ToCsv(const std::vector<RunResult>& results);
  static void PrintTable(const std::vector<RunResult>& results);
  Status WriteResults(const std::vector<RunResult>& results) const;

 private:
  Engine& engine_;
  HarnessOptions opts_;
};

// Concurrent ingest while queries run: proves the append path is not a batch only toy.
struct IngestLoadResult {
  uint64_t rows = 0;
  uint64_t bytes = 0;
  double seconds = 0;
  double bytes_per_sec() const { return seconds > 0 ? bytes / seconds : 0; }
};
IngestLoadResult RunIngestLoad(Engine& engine, const std::string& table, const std::function<RecordBatchPtr(int)>& make_batch, int batches, std::atomic<bool>& stop);

}  // namespace aster::bench
