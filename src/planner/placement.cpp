#include "aster/planner/placement.hpp"

namespace aster::planner {

PlacementMode PlacementPolicy::Choose(const Pipeline& p, size_t hbm_available) const {
  if (p.cpu) return PlacementMode::Cpu;
  if (p.pages.empty()) return PlacementMode::Resident;
  size_t resident = 0, on_host = 0, on_nvme = 0;
  for (const auto& k : p.pages) {
    const memory::PageDesc* d = nullptr;
    {
      std::lock_guard<std::recursive_mutex> lk(mm_->residency().mutex());
      d = mm_->residency().Find(k);
      if (!d) continue;
      switch (d->current_tier()) {
        case memory::Tier::Hbm: resident += d->encoded_bytes; break;
        case memory::Tier::Host: on_host += d->encoded_bytes; break;
        default: on_nvme += d->encoded_bytes; break;
      }
    }
  }
  size_t missing = on_host + on_nvme;
  if (missing == 0) return PlacementMode::Resident;
  // Everything fits: pull it in and run resident. Otherwise stream tile by tile from the slower tier.
  if (missing <= hbm_available) return PlacementMode::Resident;
  return on_nvme > on_host ? PlacementMode::StreamNvme : PlacementMode::StreamHost;
}

void PlacementPolicy::Assign(PhysicalPlan& plan) const {
  size_t available = mm_ ? mm_->hbm_free() : 0;
  for (auto& p : plan.pipelines) {
    p.placement = Choose(p, available);
    if (p.placement == PlacementMode::Resident) {
      size_t need = p.est_bytes_host_to_hbm + p.est_bytes_nvme_to_hbm;
      available = need >= available ? 0 : available - need;
    }
  }
}

}  // namespace aster::planner
