#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace aster::metrics {

// Process wide counters and per query snapshots. Every number in the design plan's metrics table is here.
struct Counter {
  std::atomic<uint64_t> value{0};
  void Add(uint64_t v) { value.fetch_add(v, std::memory_order_relaxed); }
  uint64_t Get() const { return value.load(std::memory_order_relaxed); }
};

struct QueryMetrics {
  std::string query_id;
  double wall_ms = 0;
  double kernel_ms = 0;
  double jit_compile_ms = 0;
  uint64_t bytes_hbm_to_host = 0;
  uint64_t bytes_host_to_hbm = 0;
  uint64_t bytes_nvme_to_host = 0;
  uint64_t bytes_nvme_to_hbm = 0;
  uint64_t bytes_object_to_nvme = 0;
  uint64_t bytes_estimated = 0;
  uint64_t bytes_scanned_hbm = 0;
  uint64_t rows_scanned = 0;
  uint64_t rows_output = 0;
  uint64_t segments_pruned = 0;
  uint64_t segments_scanned = 0;
  uint32_t kernel_cache_hits = 0;
  uint32_t kernel_cache_misses = 0;
  uint32_t subtrees_total = 0;
  uint32_t subtrees_cpu = 0;
  uint64_t spill_bytes = 0;
  double hbm_bandwidth_utilization = 0;
  double cost_usd = 0;

  uint64_t bytes_moved() const {
    return bytes_hbm_to_host + bytes_host_to_hbm + bytes_nvme_to_host + bytes_nvme_to_hbm + bytes_object_to_nvme;
  }
  double fallback_rate() const { return subtrees_total ? double(subtrees_cpu) / subtrees_total : 0.0; }
  double kernel_cache_hit_rate() const {
    uint32_t t = kernel_cache_hits + kernel_cache_misses;
    return t ? double(kernel_cache_hits) / t : 1.0;
  }
  std::string ToString() const;
  std::string ToJson() const;
};

class Registry {
 public:
  static Registry& Global();
  Counter& counter(const std::string& name);
  void Record(const QueryMetrics& q);
  std::vector<QueryMetrics> history() const;
  std::map<std::string, uint64_t> Snapshot() const;
  void Reset();

 private:
  mutable std::mutex mu_;
  std::map<std::string, Counter*> counters_;
  std::vector<QueryMetrics> history_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(double* out_ms) : out_(out_ms), start_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    *out_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
  }

 private:
  double* out_;
  std::chrono::steady_clock::time_point start_;
};

double CostUsd(double wall_ms, double hourly_rate_usd);

}  // namespace aster::metrics
