#include "harness.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <sstream>

#include "aster/memory/gds.hpp"

namespace aster::bench {

RunResult Harness::Run(const BenchQuery& q) {
  RunResult r;
  r.query = q.name;
  std::vector<double> times;
  metrics::QueryMetrics last;
  for (int i = 0; i < opts_.warmup + opts_.iterations; ++i) {
    plan::RelPtr rel = q.build(engine_);
    auto res = engine_.Query(rel);
    if (!res.ok()) { r.error = res.status().ToString(); return r; }
    if (i < opts_.warmup) continue;
    last = res.value().metrics;
    times.push_back(last.wall_ms);
    r.rows_out = last.rows_output;
  }
  std::sort(times.begin(), times.end());
  r.iterations = static_cast<int>(times.size());
  r.min_ms = times.front();
  r.max_ms = times.back();
  r.median_ms = times[times.size() / 2];
  r.cost_usd = metrics::CostUsd(r.median_ms, opts_.hourly_cost_usd);
  r.bytes_moved = last.bytes_moved();
  r.bytes_estimated = last.bytes_estimated;
  r.rows_scanned = last.rows_scanned;
  r.fallback_rate = last.fallback_rate();
  r.kernel_cache_hit = last.kernel_cache_hit_rate();
  r.hbm_util = last.hbm_bandwidth_utilization;
  r.segments_pruned = last.segments_pruned;
  r.segments_scanned = last.segments_scanned;
  if (opts_.verbose) std::printf("  %-24s median=%9.2f ms  cost=$%.6f  moved=%llu  est=%llu  pruned=%llu/%llu\n", r.query.c_str(), r.median_ms, r.cost_usd,
                                 (unsigned long long)r.bytes_moved, (unsigned long long)r.bytes_estimated, (unsigned long long)r.segments_pruned,
                                 (unsigned long long)(r.segments_pruned + r.segments_scanned));
  return r;
}

std::vector<RunResult> Harness::RunAll(const std::vector<BenchQuery>& queries) {
  std::vector<RunResult> out;
  for (const auto& q : queries) out.push_back(Run(q));
  return out;
}

std::string Harness::ToJson(const std::vector<RunResult>& results) {
  std::ostringstream os;
  os << "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    os << (i ? ",\n" : "") << "  {\"query\":\"" << r.query << "\",\"iterations\":" << r.iterations << ",\"min_ms\":" << r.min_ms << ",\"median_ms\":" << r.median_ms
       << ",\"max_ms\":" << r.max_ms << ",\"cost_usd\":" << r.cost_usd << ",\"bytes_moved\":" << r.bytes_moved << ",\"bytes_estimated\":" << r.bytes_estimated
       << ",\"estimate_error\":" << r.estimate_error() << ",\"rows_scanned\":" << r.rows_scanned << ",\"rows_out\":" << r.rows_out << ",\"fallback_rate\":" << r.fallback_rate
       << ",\"kernel_cache_hit\":" << r.kernel_cache_hit << ",\"hbm_util\":" << r.hbm_util << ",\"segments_pruned\":" << r.segments_pruned
       << ",\"segments_scanned\":" << r.segments_scanned << ",\"error\":\"" << r.error << "\"}";
  }
  os << "\n]\n";
  return os.str();
}

std::string Harness::ToCsv(const std::vector<RunResult>& results) {
  std::ostringstream os;
  os << "query,iterations,min_ms,median_ms,max_ms,cost_usd,bytes_moved,bytes_estimated,estimate_error,rows_scanned,rows_out,fallback_rate,kernel_cache_hit,hbm_util,segments_pruned,segments_scanned,error\n";
  for (const auto& r : results)
    os << r.query << "," << r.iterations << "," << r.min_ms << "," << r.median_ms << "," << r.max_ms << "," << r.cost_usd << "," << r.bytes_moved << "," << r.bytes_estimated << ","
       << r.estimate_error() << "," << r.rows_scanned << "," << r.rows_out << "," << r.fallback_rate << "," << r.kernel_cache_hit << "," << r.hbm_util << "," << r.segments_pruned << ","
       << r.segments_scanned << "," << r.error << "\n";
  return os.str();
}

void Harness::PrintTable(const std::vector<RunResult>& results) {
  std::printf("%-24s %10s %12s %14s %14s %8s %8s %s\n", "query", "median_ms", "cost_usd", "bytes_moved", "bytes_est", "fallback", "kcache", "status");
  double total_cost = 0;
  for (const auto& r : results) {
    total_cost += r.cost_usd;
    std::printf("%-24s %10.2f %12.6f %14llu %14llu %8.2f %8.2f %s\n", r.query.c_str(), r.median_ms, r.cost_usd, (unsigned long long)r.bytes_moved,
                (unsigned long long)r.bytes_estimated, r.fallback_rate, r.kernel_cache_hit, r.error.empty() ? "ok" : r.error.c_str());
  }
  std::printf("total cost per suite: $%.6f\n", total_cost);
}

Status Harness::WriteResults(const std::vector<RunResult>& results) const {
  if (!opts_.results_json.empty()) { std::string s = ToJson(results); ASTER_RETURN_NOT_OK(memory::WriteFile(opts_.results_json, s.data(), s.size(), false)); }
  if (!opts_.results_csv.empty()) { std::string s = ToCsv(results); ASTER_RETURN_NOT_OK(memory::WriteFile(opts_.results_csv, s.data(), s.size(), false)); }
  return Status::OK();
}

IngestLoadResult RunIngestLoad(Engine& engine, const std::string& table, const std::function<RecordBatchPtr(int)>& make_batch, int batches, std::atomic<bool>& stop) {
  IngestLoadResult r;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < batches && !stop.load(); ++i) {
    RecordBatchPtr b = make_batch(i);
    if (!b) { std::fprintf(stderr, "ingest load: generator returned no batch\n"); break; }
    auto s = engine.Append(table, b);
    if (!s.ok()) { std::fprintf(stderr, "ingest load stopped: %s\n", s.status().ToString().c_str()); break; }
    r.rows += b->num_rows();
    r.bytes += b->nbytes();
  }
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace aster::bench
