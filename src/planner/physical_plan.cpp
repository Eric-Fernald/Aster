#include "aster/planner/physical_plan.hpp"

#include <sstream>

namespace aster::planner {

const char* PlacementModeName(PlacementMode m) {
  switch (m) {
    case PlacementMode::Resident: return "resident";
    case PlacementMode::StreamHost: return "stream_host";
    case PlacementMode::StreamNvme: return "stream_nvme";
    default: return "cpu";
  }
}

std::string Pipeline::ToString() const {
  std::ostringstream os;
  os << "pipeline " << id << " [" << PlacementModeName(placement) << (fusable ? ", fused" : "") << "] ";
  for (size_t i = 0; i < ops.size(); ++i) {
    if (i) os << " -> ";
    os << plan::RelKindName(ops[i]->kind) << "#" << ops[i]->node_id;
  }
  os << " rows_in=" << est_rows_in << " rows_out=" << est_rows_out << " hbm=" << est_bytes_hbm
     << " h2d=" << est_bytes_host_to_hbm << " nvme=" << est_bytes_nvme_to_hbm << " est_s=" << est_seconds
     << " pages=" << pages.size() << " shape=" << shape_hash.substr(0, 12);
  if (!depends_on.empty()) {
    os << " after=[";
    for (size_t i = 0; i < depends_on.size(); ++i) os << (i ? "," : "") << depends_on[i];
    os << "]";
  }
  return os.str();
}

std::string PhysicalPlan::ToString() const {
  std::ostringstream os;
  os << "physical plan: pipelines=" << pipelines.size() << " scans=" << scans.size() << " pruned=" << segments_pruned << "/"
     << (segments_pruned + segments_scanned) << " est_rows=" << est_rows_scanned << " est_bytes_moved=" << est_bytes_moved()
     << " est_s=" << est_seconds << " kcache=" << kernel_cache_hits << "/" << (kernel_cache_hits + kernel_cache_misses) << "\n";
  for (const auto& p : pipelines) os << "  " << p.ToString() << "\n";
  return os.str();
}

}  // namespace aster::planner
