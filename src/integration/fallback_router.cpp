#include "aster/integration/fallback_router.hpp"

#include "aster/common/log.hpp"

namespace aster::integration {

using namespace plan;

FallbackRouter::FallbackRouter(const CapabilityRegistry& registry, const memory::BandwidthTable& bw, RoutingCosts costs)
    : registry_(registry), bw_(bw), costs_(costs) {}

void FallbackRouter::MarkInitial(const RelPtr& rel, FallbackStats& stats) const {
  for (const auto& in : rel->inputs) MarkInitial(in, stats);
  ++stats.nodes_total;
  if (rel->kind == RelKind::ExternalInput) { rel->placement = Placement::Cpu; ++stats.nodes_cpu; return; }
  std::string reason;
  Support s = registry_.CheckNode(*rel, &reason);
  if (s == Support::Gpu) {
    rel->placement = Placement::Gpu;
    ++stats.nodes_gpu;
  } else {
    rel->placement = Placement::Cpu;
    rel->fallback_reason = reason;
    stats.reasons.push_back("#" + std::to_string(rel->node_id) + " " + RelKindName(rel->kind) + ": " + reason);
    ++stats.nodes_cpu;
  }
}

void FallbackRouter::PriceRoundTrips(const RelPtr& rel, const std::function<uint64_t(const Rel&)>& rows) const {
  for (const auto& in : rel->inputs) PriceRoundTrips(in, rows);
  if (rel->placement != Placement::Cpu) return;
  // GPU capable children under a CPU parent: compare GPU time + D2H transfer against CPU time.
  for (auto& in : rel->inputs) {
    if (in->placement != Placement::Gpu || !in->fallback_reason.empty()) continue;
    uint64_t out_rows = rows(*in);
    double transfer_bytes = double(out_rows) * costs_.bytes_per_row;
    double gpu_time = 0, cpu_time = 0;
    Walk(in, [&](const RelPtr& n) {
      if (n->placement != Placement::Gpu) return;
      double r = double(rows(*n));
      gpu_time += r / costs_.gpu_rows_per_sec;
      cpu_time += r / costs_.cpu_rows_per_sec;
    });
    gpu_time += bw_.SecondsFor(memory::Tier::Hbm, memory::Tier::Host, static_cast<uint64_t>(transfer_bytes));
    if (cpu_time < gpu_time) {
      Walk(in, [&](const RelPtr& n) {
        if (n->placement == Placement::Gpu) { n->placement = Placement::Cpu; n->fallback_reason = "round trip cheaper on cpu"; }
      });
    }
  }
}

FallbackStats FallbackRouter::Route(const RelPtr& root, const std::function<uint64_t(const Rel&)>& row_estimate) const {
  FallbackStats stats;
  MarkInitial(root, stats);
  PriceRoundTrips(root, row_estimate);
  stats.nodes_cpu = 0; stats.nodes_gpu = 0;
  Walk(root, [&](const RelPtr& r) { (r->placement == Placement::Cpu ? stats.nodes_cpu : stats.nodes_gpu)++; });
  if (stats.nodes_cpu) ASTER_LOG(Info, "fallback: %u of %u nodes on cpu", stats.nodes_cpu, stats.nodes_total);
  return stats;
}

}  // namespace aster::integration
