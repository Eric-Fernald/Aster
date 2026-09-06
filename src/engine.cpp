#include "aster/engine.hpp"

#include "aster/common/log.hpp"
#include "aster/memory/gds.hpp"
#include "aster/storage/parquet_reader.hpp"
#include "aster/storage/segment.hpp"

namespace aster {

Engine::Engine(EngineConfig cfg) : cfg_(std::move(cfg)) {}

Engine::~Engine() {
  if (compactor_) compactor_->Stop();
}

Result<std::unique_ptr<Engine>> Engine::Open(const EngineConfig& cfg) {
  std::unique_ptr<Engine> e(new Engine(cfg));
  ASTER_RETURN_NOT_OK(e->Init());
  return e;
}

Status Engine::Init() {
  for (const auto& d : {cfg_.data_dir, cfg_.nvme_spill_dir, cfg_.wal_dir, cfg_.kernel_cache_dir}) ASTER_RETURN_NOT_OK(memory::EnsureDir(d));

  hal::Backend backend = cfg_.mode == HardwareMode::CpuOnly ? hal::Backend::Cpu : hal::Backend::Cuda;
  auto dev = hal::OpenDevice(backend, cfg_.device_ids.empty() ? 0 : cfg_.device_ids[0]);
  if (!dev.ok()) {
    ASTER_LOG(Warn, "device open failed (%s); falling back to cpu", dev.status().ToString().c_str());
    cfg_.mode = HardwareMode::CpuOnly;
    device_ = hal::CpuDevice();
  } else {
    device_ = dev.value();
  }

  memory::ProbeOptions po;
  po.buffer_bytes = device_->backend() == hal::Backend::Cpu ? (size_t(16) << 20) : (size_t(256) << 20);
  po.iterations = 3;
  po.nvme_probe_path = cfg_.nvme_spill_dir + "/bwprobe.bin";
  po.probe_gds = cfg_.enable_gds && device_->backend() == hal::Backend::Cuda;
  auto bw = memory::LoadOrProbe(*device_, po, cfg_.bandwidth_cache_path);
  memory::BandwidthTable table = bw.ok() ? bw.value() : memory::BandwidthTable::Defaults(cfg_.mode == HardwareMode::Coherent);
  if (!bw.ok()) ASTER_LOG(Warn, "bandwidth probe failed (%s); using defaults", bw.status().ToString().c_str());
  memory_ = std::make_unique<memory::MemoryManager>(cfg_, device_, table);

  if (!cfg_.capability_registry_path.empty()) {
    Status s = registry_.LoadTsv(cfg_.capability_registry_path);
    if (!s.ok()) ASTER_LOG(Warn, "capability registry %s: %s", cfg_.capability_registry_path.c_str(), s.ToString().c_str());
  }
  kcache_ = std::make_unique<exec::fused::KernelCache>(cfg_.kernel_cache_dir);
  jit_ = std::make_unique<exec::fused::JitCompiler>(*kcache_);

  catalog_ = std::make_unique<storage::Catalog>(cfg_.data_dir);
  wal_ = std::make_unique<ingest::WriteAheadLog>(cfg_.wal_dir);
  delta_ = std::make_unique<ingest::DeltaStore>(device_, cfg_.delta_store_max_bytes);
  ASTER_ASSIGN_OR_RETURN(recovery_, durability::Recovery::Run(*catalog_, *wal_, *delta_));
  for (const auto& table : catalog_->Tables()) {
    ASTER_ASSIGN_OR_RETURN(auto snap, catalog_->Snapshot(table));
    ASTER_ASSIGN_OR_RETURN(auto info, catalog_->GetTable(table));
    for (const auto& seg : snap->segments)
      for (auto& d : seg->Pages()) { d.dimension_hint = info.dimension_hint; memory_->RegisterPage(d); }
  }
  compactor_ = std::make_unique<ingest::Compactor>(cfg_, *catalog_, *delta_, *wal_);
  compactor_->Start();
  ingest_ = std::make_unique<ingest::IngestService>(*wal_, *delta_, compactor_.get());

  planner::PlannerContext pc;
  pc.config = &cfg_;
  pc.memory = memory_.get();
  pc.catalog = catalog_.get();
  pc.registry = &registry_;
  pc.kernel_cached = [this](const std::string& h) { return kcache_->Contains(h); };
  pc.delta_rows = [this](const std::string& t) { return delta_->Rows(t); };
  planner_ = std::make_unique<planner::Planner>(pc);

  exec::ExecutorDeps ed;
  ed.config = &cfg_;
  ed.memory = memory_.get();
  ed.catalog = catalog_.get();
  ed.jit = jit_.get();
  ed.device = device_;
  ed.delta_batches = [this](const std::string& t) { return delta_->Snapshot(t).batches; };
  executor_ = std::make_unique<exec::Executor>(ed);
  ASTER_LOG(Info, "aster engine open: mode=%s device=%s", HardwareModeName(cfg_.mode), device_->info().name.c_str());
  return Status::OK();
}

Result<planner::PhysicalPlan> Engine::Explain(const plan::RelPtr& p) { return planner_->Plan(p); }

Result<exec::QueryResult> Engine::Query(const plan::RelPtr& p, bool keep_on_device) {
  ASTER_ASSIGN_OR_RETURN(auto physical, planner_->Plan(p));
  std::string qid = "q" + std::to_string(++query_counter_);
  return executor_->Execute(physical, qid, keep_on_device);
}

Result<exec::QueryResult> Engine::QuerySubstrait(const std::string& bytes, bool keep_on_device) {
  ASTER_ASSIGN_OR_RETURN(auto rel, integration::SubstraitConsumer::FromBinary(bytes, *this));
  return Query(rel, keep_on_device);
}

Result<exec::QueryResult> Engine::QuerySubstraitJson(const std::string& json, bool keep_on_device) {
  ASTER_ASSIGN_OR_RETURN(auto rel, integration::SubstraitConsumer::FromJson(json, *this));
  return Query(rel, keep_on_device);
}

Status Engine::CreateTable(const storage::TableInfo& info) { return catalog_->CreateTable(info); }

Status Engine::DropTable(const std::string& table) {
  ASTER_ASSIGN_OR_RETURN(auto snap, catalog_->Snapshot(table));
  for (const auto& seg : snap->segments) memory_->UnregisterSegment(seg->segment_id);
  return catalog_->DropTable(table);
}

Result<uint64_t> Engine::Append(const std::string& table, RecordBatchPtr batch) {
  if (!catalog_->HasTable(table)) return Status::NotFound("table " + table);
  return ingest_->Append(table, std::move(batch));
}

Status Engine::LoadBatches(const std::string& table, const std::vector<RecordBatchPtr>& batches) {
  ASTER_ASSIGN_OR_RETURN(auto info, catalog_->GetTable(table));
  std::vector<std::shared_ptr<storage::SegmentMeta>> added;
  std::vector<RecordBatchPtr> group;
  size_t bytes = 0;
  auto flush = [&]() -> Status {
    if (group.empty()) return Status::OK();
    RecordBatchPtr merged = ConcatBatches(group);
    merged->schema = info.schema;
    SegmentId id = catalog_->NextSegmentId();
    storage::SegmentWriteOptions opts;
    opts.tile_rows = cfg_.tile_rows;
    ASTER_ASSIGN_OR_RETURN(auto meta, storage::SegmentWriter::Write(*merged, id, catalog_->SegmentPath(table, id), table, opts));
    added.push_back(std::make_shared<storage::SegmentMeta>(std::move(meta)));
    group.clear(); bytes = 0;
    return Status::OK();
  };
  for (const auto& b : batches) {
    group.push_back(b);
    bytes += b->nbytes();
    if (bytes >= cfg_.segment_target_bytes) ASTER_RETURN_NOT_OK(flush());
  }
  ASTER_RETURN_NOT_OK(flush());
  ASTER_RETURN_NOT_OK(catalog_->ReplaceSegments(table, {}, added));
  for (const auto& seg : added)
    for (auto& d : seg->Pages()) { d.dimension_hint = info.dimension_hint; memory_->RegisterPage(d); }
  return Status::OK();
}

Status Engine::ImportParquet(const std::string& table, const std::string& path) {
  ASTER_ASSIGN_OR_RETURN(auto batches, storage::ParquetReader::ReadFile(path));
  if (!catalog_->HasTable(table)) {
    ASTER_ASSIGN_OR_RETURN(Schema schema, storage::ParquetReader::ReadSchema(path));
    storage::TableInfo info;
    info.name = table;
    info.schema = schema;
    ASTER_RETURN_NOT_OK(catalog_->CreateTable(info));
  }
  return LoadBatches(table, batches);
}

Status Engine::Compact(const std::string& table) {
  ASTER_RETURN_NOT_OK(compactor_->CompactNow(table));
  ASTER_ASSIGN_OR_RETURN(auto snap, catalog_->Snapshot(table));
  ASTER_ASSIGN_OR_RETURN(auto info, catalog_->GetTable(table));
  for (const auto& seg : snap->segments)
    for (auto& d : seg->Pages()) if (!memory_->residency().Has(d.key())) { d.dimension_hint = info.dimension_hint; memory_->RegisterPage(d); }
  return Status::OK();
}

Result<Schema> Engine::TableSchema(const std::string& table) {
  ASTER_ASSIGN_OR_RETURN(auto info, catalog_->GetTable(table));
  return info.schema;
}

Result<Schema> Engine::ResolveTable(const std::vector<std::string>& names) {
  if (names.empty()) return Status::Invalid("empty table name");
  return TableSchema(names.back());
}

}  // namespace aster
