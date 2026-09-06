#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#include "aster/common/log.hpp"
#include "harness/clickbench.hpp"
#include "harness/harness.hpp"
#include "harness/tpch.hpp"

using namespace aster;

namespace {
void Usage() {
  std::fprintf(stderr,
               "usage: aster-bench --suite tpch|clickbench [--sf F] [--rows N] [--mode cpu|discrete|coherent] [--root DIR]\n"
               "                   [--iters N] [--warmup N] [--query NAME] [--ingest] [--json FILE] [--csv FILE] [--cost-per-hour USD]\n"
               "                   [--no-fusion] [--no-compressed] [--vram-mb N] [--tile-rows N] [--skip-load]\n");
}
}  // namespace

int main(int argc, char** argv) {
  std::string suite = "tpch", mode = "cpu", root = "/var/tmp/aster/bench", only;
  double sf = 0.01, cost = 3.0;
  int64_t rows = 1'000'000;
  bench::HarnessOptions ho;
  EngineConfig cfg;
  bool ingest = false, skip_load = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--suite") suite = next();
    else if (a == "--sf") sf = std::atof(next());
    else if (a == "--rows") rows = std::atoll(next());
    else if (a == "--mode") mode = next();
    else if (a == "--root") root = next();
    else if (a == "--iters") ho.iterations = std::atoi(next());
    else if (a == "--warmup") ho.warmup = std::atoi(next());
    else if (a == "--query") only = next();
    else if (a == "--ingest") ingest = true;
    else if (a == "--json") ho.results_json = next();
    else if (a == "--csv") ho.results_csv = next();
    else if (a == "--cost-per-hour") cost = std::atof(next());
    else if (a == "--no-fusion") cfg.enable_fusion = false;
    else if (a == "--no-compressed") cfg.enable_compressed_execution = false;
    else if (a == "--vram-mb") cfg.vram_budget_bytes = size_t(std::atoll(next())) << 20;
    else if (a == "--tile-rows") cfg.tile_rows = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--skip-load") skip_load = true;
    else if (a == "-v") log::SetLevel(log::Level::Debug);
    else { Usage(); return 2; }
  }
  ho.hourly_cost_usd = cost;
  cfg.mode = mode == "discrete" ? HardwareMode::Discrete : mode == "coherent" ? HardwareMode::Coherent : HardwareMode::CpuOnly;
  cfg.data_dir = root + "/data"; cfg.wal_dir = root + "/wal"; cfg.nvme_spill_dir = root + "/spill";
  cfg.kernel_cache_dir = root + "/kernels"; cfg.bandwidth_cache_path = root + "/bandwidth.tsv";
  cfg.gpu_hourly_cost_usd = cost;
  cfg.log_plans = false;
  if (mode == "cpu") log::SetLevel(log::Level::Warn);

  auto engine = Engine::Open(cfg);
  if (!engine.ok()) { std::fprintf(stderr, "%s\n", engine.status().ToString().c_str()); return 1; }
  Engine& e = *engine.value();

  std::vector<bench::BenchQuery> queries;
  if (suite == "tpch") {
    if (!skip_load) {
      std::printf("generating TPC-H SF%.3f...\n", sf);
      bench::tpch::GenOptions g; g.scale_factor = sf;
      Status s = bench::tpch::LoadInto(e, g);
      if (!s.ok()) { std::fprintf(stderr, "%s\n", s.ToString().c_str()); return 1; }
    }
    for (const auto& q : bench::tpch::Queries()) queries.push_back({q.name, q.build});
  } else if (suite == "clickbench") {
    if (!skip_load) {
      std::printf("generating hits with %lld rows...\n", (long long)rows);
      bench::clickbench::GenOptions g; g.rows = rows;
      Status s = bench::clickbench::LoadInto(e, g);
      if (!s.ok()) { std::fprintf(stderr, "%s\n", s.ToString().c_str()); return 1; }
    }
    for (const auto& q : bench::clickbench::Queries()) queries.push_back({q.name, q.build});
  } else { Usage(); return 2; }
  if (!only.empty()) {
    std::vector<bench::BenchQuery> filtered;
    for (const auto& q : queries) if (q.name == only) filtered.push_back(q);
    queries = filtered;
  }

  std::atomic<bool> stop{false};
  std::thread ingest_thread;
  bench::IngestLoadResult ingest_result;
  const std::string table = suite == "tpch" ? "lineitem" : "hits";
  if (ingest) {
    ingest_thread = std::thread([&] {
      ingest_result = bench::RunIngestLoad(e, table, [&](int i) -> RecordBatchPtr {
        RecordBatchPtr out;
        if (suite == "tpch") {
          bench::tpch::GenOptions g; g.scale_factor = 0.001; g.seed = 1000 + i;
          (void)bench::tpch::Generate(g, [&](const std::string& t, RecordBatchPtr b) { if (t == "lineitem" && !out) out = b; return Status::OK(); });
        } else {
          bench::clickbench::GenOptions g; g.rows = 50000; g.seed = 1000 + i;
          (void)bench::clickbench::Generate(g, [&](RecordBatchPtr b) { if (!out) out = b; return Status::OK(); });
        }
        return out;
      }, 1 << 20, stop);
    });
  }

  std::printf("running %zu queries (%d warmup, %d iterations)\n", queries.size(), ho.warmup, ho.iterations);
  bench::Harness harness(e, ho);
  auto results = harness.RunAll(queries);
  stop = true;
  if (ingest_thread.joinable()) ingest_thread.join();
  bench::Harness::PrintTable(results);
  if (ingest) std::printf("concurrent ingest: %llu rows, %.1f MB/s\n", (unsigned long long)ingest_result.rows, ingest_result.bytes_per_sec() / 1e6);
  Status ws = harness.WriteResults(results);
  if (!ws.ok()) std::fprintf(stderr, "%s\n", ws.ToString().c_str());
  for (const auto& r : results) if (!r.error.empty()) return 1;
  return 0;
}
