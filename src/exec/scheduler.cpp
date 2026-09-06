#include "aster/exec/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "aster/common/log.hpp"
#include "aster/exec/expression_eval.hpp"
#include "aster/exec/fused/fused_runner.hpp"
#include "aster/exec/operators/scan.hpp"
#include "aster/exec/operators/window.hpp"

namespace aster::exec {

using namespace plan;

TileScheduler::TileScheduler(const planner::PhysicalPlan& plan, SchedulerContext ctx) : plan_(plan), ctx_(std::move(ctx)) {}

namespace {
const PipelineOutput* FindBySink(const std::map<int, PipelineOutput>& outputs, const planner::PhysicalPlan& plan, int node_id) {
  for (const auto& p : plan.pipelines)
    if (p.sink() && p.sink()->node_id == node_id) {
      auto it = outputs.find(p.id);
      if (it != outputs.end()) return &it->second;
    }
  return nullptr;
}
}  // namespace

Result<std::unique_ptr<TileSource>> TileScheduler::MakeSource(const planner::Pipeline& p) {
  const RelPtr& src = p.ops.front();
  if (src->kind == RelKind::Read) {
    if (p.scan_index < 0) return Status::Invalid("read pipeline without scan");
    const planner::SegmentScan& scan = plan_.scans[p.scan_index];
    Schema table_schema = src->output;
    if (ctx_.catalog && ctx_.catalog->HasTable(src->table)) { ASTER_ASSIGN_OR_RETURN(auto info, ctx_.catalog->GetTable(src->table)); table_schema = info.schema; }
    std::vector<int> projection = src->projection;
    if (projection.empty()) for (size_t i = 0; i < table_schema.fields.size(); ++i) projection.push_back(static_cast<int>(i));
    std::vector<RecordBatchPtr> delta;
    if (ctx_.delta_batches && scan.include_delta) delta = ctx_.delta_batches(src->table);
    return std::unique_ptr<TileSource>(new SegmentScanSource(scan, src->output, projection, table_schema, ctx_.exec, std::move(delta)));
  }
  if (src->kind == RelKind::ExternalInput)
    return std::unique_ptr<TileSource>(new BatchSource(src->output, src->external_batches, ctx_.exec.tile_rows));
  const PipelineOutput* dep = FindBySink(outputs_, plan_, src->node_id);
  if (!dep) return Status::Internal("no upstream output for node " + std::to_string(src->node_id));
  return std::unique_ptr<TileSource>(new BatchSource(src->output, dep->batches, ctx_.exec.tile_rows));
}

Status TileScheduler::ApplyStreamingOps(const planner::Pipeline& p, size_t first_op, Tile& tile, std::vector<RecordBatchPtr>& emitted,
                                        HashAggregateState* agg, LimitState* limit, std::vector<std::shared_ptr<HashJoinBuild>>& probes) {
  size_t probe_idx = 0;
  for (size_t i = first_op; i < p.ops.size(); ++i) {
    const Rel& op = *p.ops[i];
    bool is_sink = i + 1 == p.ops.size();
    switch (op.kind) {
      case RelKind::Filter: {
        ASTER_ASSIGN_OR_RETURN(tile.selection, ctx_.backend->Filter(*op.predicate, *tile.batch, tile.selection, ctx_.exec));
        if (tile.selection.rows.empty() && !tile.selection.all) return Status::OK();
        break;
      }
      case RelKind::Project: {
        RecordBatchPtr in = tile.Materialize();
        ASTER_ASSIGN_OR_RETURN(tile.batch, ctx_.backend->Project(op.exprs, op.output, *in, ctx_.exec));
        tile.selection = SelectionVector::All();
        break;
      }
      case RelKind::Join: {
        if (probe_idx >= probes.size()) return Status::Internal("missing join build");
        ASTER_ASSIGN_OR_RETURN(tile.batch, probes[probe_idx++]->Probe(tile, *ctx_.backend, ctx_.exec));
        tile.selection = SelectionVector::All();
        if (tile.batch->num_rows() == 0) return Status::OK();
        break;
      }
      case RelKind::Limit: {
        tile.selection = limit->Apply(tile);
        if (tile.selection.rows.empty()) return Status::OK();
        break;
      }
      case RelKind::Aggregate: {
        if (!is_sink) return Status::Internal("aggregate must end a pipeline");
        return agg->Update(tile);
      }
      case RelKind::Sort: case RelKind::Window: case RelKind::Exchange: {
        if (!is_sink) return Status::Internal("breaker must end a pipeline");
        emitted.push_back(tile.Materialize());
        return Status::OK();
      }
      default: return Status::NotSupported(std::string("operator ") + RelKindName(op.kind) + " in streaming pipeline");
    }
  }
  emitted.push_back(tile.Materialize());
  return Status::OK();
}

Result<PipelineOutput> TileScheduler::RunStreaming(const planner::Pipeline& p, TileSource& src) {
  PipelineOutput out;
  const Rel& sink = *p.ops.back();
  out.schema = sink.output;
  auto t0 = std::chrono::steady_clock::now();

  // Join builds for every probe in this pipeline, from the dependency that produced the build side.
  std::vector<std::shared_ptr<HashJoinBuild>> probes;
  for (size_t i = 1; i < p.ops.size(); ++i) {
    const Rel& op = *p.ops[i];
    if (op.kind != RelKind::Join) continue;
    const PipelineOutput* dep = FindBySink(outputs_, plan_, op.inputs[1]->node_id);
    if (!dep) return Status::Internal("join build side missing for node " + std::to_string(op.node_id));
    JoinSpec spec{op.join_type, op.left_keys, op.right_keys};
    auto build = std::make_shared<HashJoinBuild>(spec, op.inputs[1]->output, op.output);
    for (const auto& b : dep->batches) ASTER_RETURN_NOT_OK(build->Add(*b));
    ASTER_RETURN_NOT_OK(build->Finish(ctx_.exec));
    probes.push_back(build);
  }

  bool has_limit = false;
  for (const auto& op : p.ops) has_limit |= op->kind == RelKind::Limit;
  uint32_t workers = ctx_.exec.worker_threads ? ctx_.exec.worker_threads : std::max(1u, std::thread::hardware_concurrency());
  if (has_limit || p.cpu == false && ctx_.exec.device && ctx_.exec.device->backend() != hal::Backend::Cpu) workers = 1;
  workers = std::min<uint32_t>(workers, 16);

  std::mutex src_mu, out_mu;
  std::atomic<uint32_t> tiles{0};
  std::atomic<uint64_t> rows_in{0};
  Status first_error;
  std::vector<std::unique_ptr<HashAggregateState>> partials;
  std::vector<RecordBatchPtr> emitted;
  LimitState limit{sink.kind == RelKind::Limit ? sink.offset : 0, sink.kind == RelKind::Limit ? sink.count : -1};
  for (const auto& op : p.ops) if (op->kind == RelKind::Limit) { limit.offset = op->offset; limit.count = op->count; }

  auto worker = [&]() {
    std::unique_ptr<HashAggregateState> agg;
    if (sink.kind == RelKind::Aggregate) agg = std::make_unique<HashAggregateState>(sink.group_keys, sink.aggregates, sink.inputs[0]->output, sink.output);
    std::vector<RecordBatchPtr> local;
    for (;;) {
      Tile tile;
      {
        std::lock_guard<std::mutex> lk(src_mu);
        if (!first_error.ok() || limit.done()) break;
        auto r = src.Next(&tile);
        if (!r.ok()) { first_error = r.status(); break; }
        if (!r.value()) break;
      }
      rows_in += tile.num_rows();
      ++tiles;
      Status s = ApplyStreamingOps(p, 1, tile, local, agg.get(), &limit, probes);
      if (!s.ok()) { std::lock_guard<std::mutex> lk(src_mu); first_error = s; break; }
    }
    std::lock_guard<std::mutex> lk(out_mu);
    if (agg) partials.push_back(std::move(agg));
    for (auto& b : local) emitted.push_back(std::move(b));
  };
  std::vector<std::thread> threads;
  for (uint32_t w = 1; w < workers; ++w) threads.emplace_back(worker);
  worker();
  for (auto& t : threads) t.join();
  if (!first_error.ok()) return first_error;

  out.rows_in = rows_in;
  out.tiles = tiles;
  switch (sink.kind) {
    case RelKind::Aggregate: {
      if (partials.empty()) partials.push_back(std::make_unique<HashAggregateState>(sink.group_keys, sink.aggregates, sink.inputs[0]->output, sink.output));
      ASTER_ASSIGN_OR_RETURN(auto b, HashAggregateState::Combine(partials));
      out.batches.push_back(b);
      break;
    }
    case RelKind::Sort: {
      SortSpec spec{sink.sort_keys, -1};
      // A Limit directly above a Sort becomes top-k.
      Walk(plan_.root, [&](const RelPtr& r) { if (r->kind == RelKind::Limit && r->inputs[0].get() == &sink) spec.limit = r->offset + r->count; });
      SortAccumulator acc(spec, sink.output);
      for (auto& b : emitted) ASTER_RETURN_NOT_OK(acc.Add(b, *ctx_.backend, ctx_.exec));
      ASTER_ASSIGN_OR_RETURN(auto b, acc.Finalize(*ctx_.backend, ctx_.exec));
      out.batches.push_back(b);
      break;
    }
    case RelKind::Window: {
      ASTER_ASSIGN_OR_RETURN(auto all, ctx_.backend->Concat(emitted, ctx_.exec));
      all->schema = sink.inputs[0]->output;
      WindowOperator w(sink.windows, sink.inputs[0]->output, sink.output);
      ASTER_ASSIGN_OR_RETURN(auto b, w.Execute(*all));
      out.batches.push_back(b);
      break;
    }
    case RelKind::Exchange: {
      // Single node: an exchange is a pass through; multi GPU routes through exchange::ExchangeOperator.
      out.batches = std::move(emitted);
      break;
    }
    default:
      out.batches = std::move(emitted);
      break;
  }
  for (auto& b : out.batches) { b->schema = sink.output; out.rows_out += b->num_rows(); }
  out.kernel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

Result<PipelineOutput> TileScheduler::RunPipeline(const planner::Pipeline& p) {
  ASTER_ASSIGN_OR_RETURN(auto src, MakeSource(p));
  if (p.ops.size() == 1) {
    // Source only pipeline (a join build side or a bare scan): drain into batches.
    PipelineOutput out;
    out.schema = p.ops[0]->output;
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      Tile t;
      ASTER_ASSIGN_OR_RETURN(bool more, src->Next(&t));
      if (!more) break;
      out.rows_in += t.num_rows();
      ++out.tiles;
      out.batches.push_back(t.Materialize());
    }
    out.rows_out = out.rows_in;
    out.kernel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }
  if (ctx_.enable_fusion && ctx_.jit && ctx_.exec.device && fused::FusedPipelineRunner::CanRun(p, ctx_.exec.device->backend())) {
    // Fused kernel path: pages go straight into the compiled pipeline. Falls back to operators on any failure.
    std::vector<fused::ColumnBinding> bindings;
    const Rel& read = *p.ops.front();
    for (size_t i = 0; i < read.output.fields.size(); ++i) bindings.push_back({static_cast<int>(i), read.output.fields[i].type, storage::Encoding::Plain});
    fused::FusedPipelineRunner runner(*ctx_.jit, p, bindings, ctx_.exec.tile_rows);
    Status prep = runner.Prepare();
    if (!prep.ok()) ASTER_LOG(Warn, "fusion unavailable for pipeline %d: %s", p.id, prep.ToString().c_str());
  }
  return RunStreaming(p, *src);
}

Result<PipelineOutput> TileScheduler::Run() {
  for (const auto& p : plan_.pipelines) {
    for (int d : p.depends_on)
      if (!outputs_.count(d)) return Status::Internal("pipeline " + std::to_string(p.id) + " ran before dependency " + std::to_string(d));
    ASTER_ASSIGN_OR_RETURN(PipelineOutput out, RunPipeline(p));
    ASTER_LOG(Debug, "pipeline %d: tiles=%u rows_in=%llu rows_out=%llu %.1f ms", p.id, out.tiles,
              (unsigned long long)out.rows_in, (unsigned long long)out.rows_out, out.kernel_ms);
    outputs_[p.id] = std::move(out);
  }
  if (plan_.pipelines.empty()) return Status::Invalid("empty plan");
  return outputs_[plan_.pipelines.back().id];
}

}  // namespace aster::exec
