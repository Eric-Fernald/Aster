#pragma once
#include <string>
#include <vector>

#include "aster/integration/capability_registry.hpp"
#include "aster/integration/plan_ir.hpp"
#include "aster/memory/bandwidth_probe.hpp"

namespace aster::integration {

struct FallbackStats {
  uint32_t nodes_total = 0;
  uint32_t nodes_cpu = 0;
  uint32_t nodes_gpu = 0;
  std::vector<std::string> reasons;
  double fallback_rate() const { return nodes_total ? double(nodes_cpu) / nodes_total : 0.0; }
};

struct RoutingCosts {
  double gpu_rows_per_sec = 5e9;   // throughput used to price a node on each side
  double cpu_rows_per_sec = 2e8;
  double bytes_per_row = 32;
};

// Per subtree placement. A node the registry cannot run on GPU goes to CPU; its output becomes an
// external Arrow input for the GPU parent. The round trip is priced so a GPU capable child sitting
// under a CPU parent may be pulled to CPU when the transfer costs more than the speedup.
class FallbackRouter {
 public:
  FallbackRouter(const CapabilityRegistry& registry, const memory::BandwidthTable& bw, RoutingCosts costs = {});
  FallbackStats Route(const plan::RelPtr& root, const std::function<uint64_t(const plan::Rel&)>& row_estimate) const;

 private:
  void MarkInitial(const plan::RelPtr& rel, FallbackStats& stats) const;
  void PriceRoundTrips(const plan::RelPtr& rel, const std::function<uint64_t(const plan::Rel&)>& rows) const;
  const CapabilityRegistry& registry_;
  const memory::BandwidthTable& bw_;
  RoutingCosts costs_;
};

}  // namespace aster::integration
