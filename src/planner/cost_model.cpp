#include "aster/planner/cost_model.hpp"

#include <algorithm>
#include <cmath>

namespace aster::planner {

using namespace plan;

CostModel::CostModel(const memory::BandwidthTable& bw, const memory::ResidencyMap* residency, CostParams params)
    : bw_(bw), residency_(residency), params_(params) {}

double CostModel::Selectivity(const Expr& pred) const {
  if (pred.kind == ExprKind::Literal) {
    if (auto b = std::get_if<bool>(&pred.literal)) return *b ? 1.0 : 0.0;
    return 1.0;
  }
  if (pred.kind != ExprKind::Call) return params_.default_selectivity;
  const std::string& f = pred.function;
  if (f == "and") { double s = 1.0; for (const auto& a : pred.args) s *= Selectivity(*a); return s; }
  if (f == "or") { double s = 0.0; for (const auto& a : pred.args) s = s + Selectivity(*a) - s * Selectivity(*a); return s; }
  if (f == "not") return pred.args.empty() ? 0.5 : 1.0 - Selectivity(*pred.args[0]);
  if (f == "equal" || f == "in" || f == "is_null") return params_.equality_selectivity;
  if (f == "lt" || f == "lte" || f == "gt" || f == "gte") return params_.range_selectivity;
  if (f == "between") return params_.range_selectivity * 0.5;
  if (f == "like" || f == "starts_with" || f == "ends_with" || f == "contains") return params_.like_selectivity;
  if (f == "is_not_null" || f == "not_equal") return 0.9;
  return params_.default_selectivity;
}

uint64_t CostModel::EstimateRows(const Rel& rel, const std::function<uint64_t(const Rel&)>& source_rows) const {
  auto child = [&](size_t i) -> uint64_t { return i < rel.inputs.size() ? EstimateRows(*rel.inputs[i], source_rows) : 0; };
  switch (rel.kind) {
    case RelKind::Read: {
      uint64_t r = source_rows(rel);
      return rel.pushed_filter ? static_cast<uint64_t>(r * Selectivity(*rel.pushed_filter)) : r;
    }
    case RelKind::ExternalInput: {
      uint64_t n = 0;
      for (const auto& b : rel.external_batches) n += b->num_rows();
      return n;
    }
    case RelKind::Filter: return static_cast<uint64_t>(child(0) * Selectivity(*rel.predicate));
    case RelKind::Project: case RelKind::Window: case RelKind::Sort: case RelKind::Exchange: return child(0);
    case RelKind::Limit: {
      uint64_t c = child(0);
      return rel.count >= 0 ? std::min<uint64_t>(c, static_cast<uint64_t>(rel.count)) : c;
    }
    case RelKind::Join: {
      uint64_t l = child(0), r = child(1);
      switch (rel.join_type) {
        case JoinType::Semi: return l / 2;
        case JoinType::Anti: return l / 2;
        case JoinType::Left: return std::max(l, static_cast<uint64_t>(std::sqrt(double(l) * double(r))));
        default: return rel.left_keys.empty() ? l * r : std::max<uint64_t>(1, std::max(l, r));
      }
    }
    case RelKind::Aggregate: {
      uint64_t c = child(0);
      if (rel.group_keys.empty()) return 1;
      double groups = std::pow(double(std::max<uint64_t>(c, 1)), 0.5) * rel.group_keys.size();
      return static_cast<uint64_t>(std::min(double(c), groups));
    }
  }
  return 0;
}

void CostModel::EstimatePipeline(Pipeline& p, const std::vector<SegmentScan>& scans, bool kernel_cached) const {
  p.est_bytes_hbm = 0;
  p.est_bytes_host_to_hbm = 0;
  p.est_bytes_nvme_to_hbm = 0;
  if (p.scan_index >= 0 && p.scan_index < static_cast<int>(scans.size())) {
    const SegmentScan& scan = scans[p.scan_index];
    for (const auto& seg : scan.segments) {
      for (ColumnId c : scan.columns) {
        if (c >= seg->columns.size()) continue;
        uint64_t bytes = seg->columns[c].bytes;
        p.est_bytes_hbm += bytes;
        memory::Tier t = residency_ ? residency_->TierOf({seg->segment_id, c}) : memory::Tier::Nvme;
        if (t == memory::Tier::Host) p.est_bytes_host_to_hbm += bytes;
        else if (t != memory::Tier::Hbm) p.est_bytes_nvme_to_hbm += bytes;
      }
    }
  }
  double seconds = 0;
  seconds += bw_.SecondsFor(memory::Tier::Host, memory::Tier::Hbm, p.est_bytes_host_to_hbm);
  seconds += bw_.SecondsFor(memory::Tier::Nvme, memory::Tier::Hbm, p.est_bytes_nvme_to_hbm);
  seconds += double(p.est_bytes_hbm) / params_.hbm_bytes_per_sec;
  double rows_per_sec = p.cpu ? params_.cpu_rows_per_sec : params_.gpu_rows_per_sec;
  seconds += double(p.est_rows_in) * std::max<size_t>(1, p.ops.size()) / rows_per_sec;
  if (p.fusable && !kernel_cached) seconds += params_.jit_compile_seconds;
  p.est_seconds = seconds;
}

}  // namespace aster::planner
