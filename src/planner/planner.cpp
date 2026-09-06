#include "aster/planner/planner.hpp"

#include <set>

#include "aster/common/log.hpp"
#include "aster/planner/pipeline_splitter.hpp"
#include "aster/planner/placement.hpp"
#include "aster/planner/prefetch_schedule.hpp"
#include "aster/planner/segment_pruner.hpp"

namespace aster::planner {

using namespace plan;

Planner::Planner(PlannerContext ctx) : ctx_(std::move(ctx)) {}

namespace {
// Physical columns a Read must materialize: its projection restricted to what parents reference.
void ReferencedFields(const Expr& e, std::set<int>& out) {
  if (e.is_column()) out.insert(e.field_index);
  for (const auto& a : e.args) ReferencedFields(*a, out);
}
}  // namespace

Status Planner::PruneScans(PhysicalPlan& plan) {
  for (const auto& read : Collect(plan.root, RelKind::Read)) {
    SegmentScan scan;
    scan.read_node_id = read->node_id;
    scan.table = read->table;
    for (size_t i = 0; i < read->projection.size(); ++i) scan.columns.push_back(static_cast<ColumnId>(i));
    if (ctx_.catalog && ctx_.catalog->HasTable(read->table)) {
      ASTER_ASSIGN_OR_RETURN(auto snap, ctx_.catalog->Snapshot(read->table));
      ASTER_ASSIGN_OR_RETURN(auto info, ctx_.catalog->GetTable(read->table));
      ExprPtr pred = SegmentPruner::CollectPredicates(read, plan.root);
      PruneResult pr = SegmentPruner::Prune(snap->segments, pred.get(), info.schema, read->projection);
      scan.segments = pr.kept;
      scan.rows = pr.kept_rows;
      scan.pruned_segments = pr.pruned_segments;
      scan.pruned_rows = pr.pruned_rows;
      for (const auto& s : scan.segments)
        for (ColumnId c : scan.columns) {
          int ci = s->ColumnIndex(info.schema.fields.at(read->projection[c]).name);
          if (ci >= 0) scan.bytes += s->columns[ci].bytes;
        }
      plan.segments_pruned += pr.pruned_segments;
      plan.segments_scanned += pr.kept.size();
    }
    if (ctx_.delta_rows) scan.rows += ctx_.delta_rows(read->table);
    plan.est_rows_scanned += scan.rows;
    plan.scans.push_back(std::move(scan));
  }
  return Status::OK();
}

uint64_t Planner::SourceRows(const Rel& rel, const PhysicalPlan& plan) const {
  for (const auto& s : plan.scans)
    if (s.read_node_id == rel.node_id) return s.rows;
  return 0;
}

void Planner::EstimateAndPlace(PhysicalPlan& plan) {
  const memory::BandwidthTable& bw = ctx_.memory ? ctx_.memory->bandwidth() : memory::BandwidthTable::Defaults(false);
  CostModel cost(bw, ctx_.memory ? &ctx_.memory->residency() : nullptr);
  auto rows_of = [&](const Rel& r) { return SourceRows(r, plan); };

  for (auto& p : plan.pipelines) {
    const Rel* src = p.source();
    if (src && src->kind == RelKind::Read) {
      for (size_t i = 0; i < plan.scans.size(); ++i)
        if (plan.scans[i].read_node_id == src->node_id) p.scan_index = static_cast<int>(i);
      if (p.scan_index >= 0) {
        const SegmentScan& scan = plan.scans[p.scan_index];
        std::set<int> used;
        for (const auto& op : p.ops) {
          if (op->predicate) ReferencedFields(*op->predicate, used);
          for (const auto& e : op->exprs) ReferencedFields(*e, used);
          for (const auto& k : op->group_keys) ReferencedFields(*k, used);
          for (const auto& a : op->aggregates) for (const auto& e : a.args) ReferencedFields(*e, used);
          for (int k : op->left_keys) used.insert(k);
        }
        for (const auto& seg : scan.segments)
          for (ColumnId c : scan.columns) {
            if (!used.empty() && !used.count(static_cast<int>(c)) && p.ops.size() > 1) continue;
            const std::string& name = src->output.fields[c].name;
            int ci = seg->ColumnIndex(name);
            if (ci >= 0) p.pages.push_back({seg->segment_id, static_cast<ColumnId>(ci)});
          }
      }
    }
    p.est_rows_in = src ? cost.EstimateRows(*src, rows_of) : 0;
    p.est_rows_out = p.sink() ? cost.EstimateRows(*p.sink(), rows_of) : 0;
    bool cached = ctx_.kernel_cached ? ctx_.kernel_cached(p.shape_hash) : false;
    if (p.fusable) (cached ? plan.kernel_cache_hits : plan.kernel_cache_misses)++;
    cost.EstimatePipeline(p, plan.scans, cached);
    plan.est_bytes_hbm += p.est_bytes_hbm;
    plan.est_bytes_host_to_hbm += p.est_bytes_host_to_hbm;
    plan.est_bytes_nvme_to_hbm += p.est_bytes_nvme_to_hbm;
    plan.est_seconds += p.est_seconds;
  }
  if (ctx_.memory) {
    PlacementPolicy placement(ctx_.memory, cost);
    placement.Assign(plan);
  } else {
    for (auto& p : plan.pipelines) p.placement = p.cpu ? PlacementMode::Cpu : PlacementMode::Resident;
  }
}

Result<PhysicalPlan> Planner::Plan(const RelPtr& root) {
  if (!root) return Status::Invalid("null plan");
  AssignNodeIds(root);
  PhysicalPlan plan;
  plan.root = root;

  ASTER_RETURN_NOT_OK(PruneScans(plan));

  const memory::BandwidthTable& bw = ctx_.memory ? ctx_.memory->bandwidth() : memory::BandwidthTable::Defaults(false);
  integration::CapabilityRegistry defaults;
  const integration::CapabilityRegistry& reg = ctx_.registry ? *ctx_.registry : defaults;
  if (ctx_.config && ctx_.config->mode == HardwareMode::CpuOnly) {
    Walk(root, [](const RelPtr& r) { r->placement = Placement::Cpu; r->fallback_reason = "cpu only mode"; });
    fallback_ = {};
    Walk(root, [&](const RelPtr&) { ++fallback_.nodes_total; ++fallback_.nodes_cpu; });
  } else {
    integration::FallbackRouter router(reg, bw);
    CostModel cost(bw, nullptr);
    fallback_ = router.Route(root, [&](const Rel& r) { return cost.EstimateRows(r, [&](const Rel& s) { return SourceRows(s, plan); }); });
  }

  plan.pipelines = PipelineSplitter::Split(root);
  EstimateAndPlace(plan);

  uint32_t depth = ctx_.config ? ctx_.config->prefetch_depth : 4;
  plan.prefetch_order = PrefetchScheduler(depth).Schedule(plan);
  if (!ctx_.config || ctx_.config->log_plans) ASTER_LOG(Info, "%s", plan.ToString().c_str());
  return plan;
}

}  // namespace aster::planner
