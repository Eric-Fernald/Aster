#include "aster/metrics/metrics.hpp"

#include <sstream>

namespace aster::metrics {

Registry& Registry::Global() {
  static Registry r;
  return r;
}

Counter& Registry::counter(const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = counters_.find(name);
  if (it == counters_.end()) it = counters_.emplace(name, new Counter()).first;
  return *it->second;
}

void Registry::Record(const QueryMetrics& q) {
  std::lock_guard<std::mutex> lk(mu_);
  history_.push_back(q);
}

std::vector<QueryMetrics> Registry::history() const {
  std::lock_guard<std::mutex> lk(mu_);
  return history_;
}

std::map<std::string, uint64_t> Registry::Snapshot() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::map<std::string, uint64_t> out;
  for (const auto& [k, v] : counters_) out[k] = v->Get();
  return out;
}

void Registry::Reset() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [k, v] : counters_) v->value.store(0);
  history_.clear();
}

double CostUsd(double wall_ms, double hourly_rate_usd) { return wall_ms / 3600000.0 * hourly_rate_usd; }

std::string QueryMetrics::ToString() const {
  std::ostringstream os;
  os << "query=" << query_id << " wall_ms=" << wall_ms << " kernel_ms=" << kernel_ms << " jit_ms=" << jit_compile_ms
     << " bytes_moved=" << bytes_moved() << " bytes_est=" << bytes_estimated << " rows_scanned=" << rows_scanned
     << " rows_out=" << rows_output << " pruned=" << segments_pruned << "/" << (segments_pruned + segments_scanned)
     << " kcache_hit=" << kernel_cache_hit_rate() << " fallback=" << fallback_rate() << " spill=" << spill_bytes
     << " hbm_util=" << hbm_bandwidth_utilization << " cost_usd=" << cost_usd;
  return os.str();
}

std::string QueryMetrics::ToJson() const {
  std::ostringstream os;
  os << "{\"query_id\":\"" << query_id << "\",\"wall_ms\":" << wall_ms << ",\"kernel_ms\":" << kernel_ms
     << ",\"jit_compile_ms\":" << jit_compile_ms << ",\"bytes_hbm_to_host\":" << bytes_hbm_to_host
     << ",\"bytes_host_to_hbm\":" << bytes_host_to_hbm << ",\"bytes_nvme_to_host\":" << bytes_nvme_to_host
     << ",\"bytes_nvme_to_hbm\":" << bytes_nvme_to_hbm << ",\"bytes_object_to_nvme\":" << bytes_object_to_nvme
     << ",\"bytes_estimated\":" << bytes_estimated << ",\"bytes_scanned_hbm\":" << bytes_scanned_hbm
     << ",\"rows_scanned\":" << rows_scanned << ",\"rows_output\":" << rows_output
     << ",\"segments_pruned\":" << segments_pruned << ",\"segments_scanned\":" << segments_scanned
     << ",\"kernel_cache_hit_rate\":" << kernel_cache_hit_rate() << ",\"fallback_rate\":" << fallback_rate()
     << ",\"spill_bytes\":" << spill_bytes << ",\"hbm_bandwidth_utilization\":" << hbm_bandwidth_utilization
     << ",\"cost_usd\":" << cost_usd << "}";
  return os.str();
}

}  // namespace aster::metrics
