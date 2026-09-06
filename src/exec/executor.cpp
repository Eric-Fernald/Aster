#include "aster/exec/executor.hpp"

#include <chrono>

#include "aster/common/log.hpp"
#include "aster/durability/fault_injector.hpp"

namespace aster::exec {

Executor::Executor(ExecutorDeps deps) : deps_(std::move(deps)) {
  hal::Backend b = deps_.device ? deps_.device->backend() : hal::Backend::Cpu;
  backend_ = MakeBackend(b, deps_.config ? deps_.config->prefer_cudf_operators : true);
}

Result<QueryResult> Executor::Execute(const planner::PhysicalPlan& plan, const std::string& query_id, bool keep_on_device) {
  QueryResult result;
  result.metrics.query_id = query_id;
  auto t0 = std::chrono::steady_clock::now();
  memory::MemoryStats before = deps_.memory ? deps_.memory->stats() : memory::MemoryStats{};

  std::vector<memory::PageKey> all_pages;
  for (const auto& p : plan.pipelines) all_pages.insert(all_pages.end(), p.pages.begin(), p.pages.end());
  if (deps_.memory) {
    deps_.memory->MarkScheduled(all_pages, true);
    deps_.memory->Prefetch(plan.prefetch_order);
  }

  SchedulerContext sctx;
  sctx.exec.query_id = next_query_id_++;
  sctx.exec.tile_rows = deps_.config ? deps_.config->tile_rows : 32 * 1024;
  sctx.exec.memory = deps_.memory;
  sctx.exec.device = deps_.device.get();
  sctx.exec.compressed_execution = deps_.config ? deps_.config->enable_compressed_execution : true;
  sctx.backend = backend_.get();
  sctx.jit = deps_.jit;
  sctx.catalog = deps_.catalog;
  sctx.delta_batches = deps_.delta_batches;
  sctx.enable_fusion = deps_.config ? deps_.config->enable_fusion : true;
  if (deps_.device && deps_.device->backend() != hal::Backend::Cpu) {
    auto s = deps_.device->CreateStream();
    if (s.ok()) sctx.exec.stream = s.value();
  }
  uint64_t rows_scanned = 0, bytes_scanned = 0;
  sctx.exec.on_scan = [&](uint64_t r, uint64_t b) { rows_scanned += r; bytes_scanned += b; };

  ASTER_FAULT_POINT("query.before_run");
  TileScheduler sched(plan, sctx);
  auto run = sched.Run();
  if (deps_.memory) {
    deps_.memory->MarkScheduled(all_pages, false);
    deps_.memory->device_pool().ReleaseQuery(sctx.exec.query_id);
  }
  if (sctx.exec.stream.handle && deps_.device) deps_.device->DestroyStream(sctx.exec.stream);
  if (!run.ok()) return run.status();
  PipelineOutput& out = run.value();
  ASTER_FAULT_POINT("query.after_run");

  result.schema = out.schema;
  result.batches = std::move(out.batches);
  if (keep_on_device && deps_.device && deps_.device->backend() != hal::Backend::Cpu) {
    for (auto& b : result.batches)
      for (auto& c : b->columns) { ASTER_ASSIGN_OR_RETURN(c, deps_.device->ColumnToDevice(c)); }
  }

  metrics::QueryMetrics& m = result.metrics;
  m.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  double kernel = 0, jit = 0;
  for (const auto& [id, po] : sched.outputs()) { kernel += po.kernel_ms; jit += po.jit_ms; }
  m.kernel_ms = kernel;
  m.jit_compile_ms = jit;
  m.rows_scanned = rows_scanned;
  m.bytes_scanned_hbm = bytes_scanned;
  m.rows_output = static_cast<uint64_t>(result.num_rows());
  m.segments_pruned = plan.segments_pruned;
  m.segments_scanned = plan.segments_scanned;
  m.bytes_estimated = plan.est_bytes_moved();
  m.kernel_cache_hits = plan.kernel_cache_hits;
  m.kernel_cache_misses = plan.kernel_cache_misses;
  if (deps_.memory) {
    memory::MemoryStats after = deps_.memory->stats();
    m.bytes_nvme_to_host = after.bytes_nvme_to_host - before.bytes_nvme_to_host;
    m.bytes_nvme_to_hbm = after.bytes_nvme_to_hbm - before.bytes_nvme_to_hbm;
    m.bytes_host_to_hbm = after.bytes_host_to_hbm - before.bytes_host_to_hbm;
    m.bytes_hbm_to_host = after.bytes_hbm_to_host - before.bytes_hbm_to_host;
    m.spill_bytes = deps_.memory->spill().total_spilled();
    double hbm_bps = deps_.memory->bandwidth().BytesPerSec(memory::Tier::Hbm, memory::Tier::Hbm);
    if (kernel > 0 && hbm_bps > 0) m.hbm_bandwidth_utilization = std::min(1.0, (double(bytes_scanned) / (kernel / 1000.0)) / hbm_bps);
  }
  Walk(plan.root, [&](const plan::RelPtr& r) { ++m.subtrees_total; if (r->placement == plan::Placement::Cpu) ++m.subtrees_cpu; });
  double rate = deps_.config ? (deps_.device && deps_.device->backend() != hal::Backend::Cpu ? deps_.config->gpu_hourly_cost_usd : deps_.config->cpu_hourly_cost_usd) : 1.0;
  m.cost_usd = metrics::CostUsd(m.wall_ms, rate);
  metrics::Registry::Global().Record(m);
  ASTER_LOG(Info, "%s", m.ToString().c_str());
  return result;
}

}  // namespace aster::exec
