#pragma once
#include <memory>
#include <string>
#include <vector>

#include "aster/common/config.hpp"
#include "aster/common/status.hpp"
#include "aster/durability/recovery.hpp"
#include "aster/exec/executor.hpp"
#include "aster/exec/fused/jit_compiler.hpp"
#include "aster/exec/fused/kernel_cache.hpp"
#include "aster/hal/device.hpp"
#include "aster/ingest/compactor.hpp"
#include "aster/ingest/delta_store.hpp"
#include "aster/ingest/ingest_service.hpp"
#include "aster/ingest/wal.hpp"
#include "aster/integration/capability_registry.hpp"
#include "aster/integration/plan_ir.hpp"
#include "aster/integration/substrait_consumer.hpp"
#include "aster/memory/memory_manager.hpp"
#include "aster/planner/planner.hpp"
#include "aster/storage/catalog.hpp"

namespace aster {

// The engine facade: not a database. Hosts hand it Substrait plans (or IR built with PlanBuilder)
// and get Arrow layout batches back, on host or left on device for DLPack.
class Engine : public integration::TableResolver {
 public:
  static Result<std::unique_ptr<Engine>> Open(const EngineConfig& cfg);
  ~Engine();

  Result<exec::QueryResult> Query(const plan::RelPtr& plan, bool keep_on_device = false);
  Result<exec::QueryResult> QuerySubstrait(const std::string& plan_bytes, bool keep_on_device = false);
  Result<exec::QueryResult> QuerySubstraitJson(const std::string& plan_json, bool keep_on_device = false);
  Result<planner::PhysicalPlan> Explain(const plan::RelPtr& plan);

  Status CreateTable(const storage::TableInfo& info);
  Status DropTable(const std::string& table);
  Result<uint64_t> Append(const std::string& table, RecordBatchPtr batch);
  Status ImportParquet(const std::string& table, const std::string& path);
  Status LoadBatches(const std::string& table, const std::vector<RecordBatchPtr>& batches);  // direct to segments
  Status Compact(const std::string& table);
  Result<Schema> ResolveTable(const std::vector<std::string>& names) override;
  Result<Schema> TableSchema(const std::string& table);

  const EngineConfig& config() const { return cfg_; }
  hal::Device& device() { return *device_; }
  memory::MemoryManager& memory() { return *memory_; }
  storage::Catalog& catalog() { return *catalog_; }
  integration::CapabilityRegistry& registry() { return registry_; }
  ingest::IngestService& ingest() { return *ingest_; }
  ingest::DeltaStore& delta() { return *delta_; }
  ingest::Compactor& compactor() { return *compactor_; }
  exec::fused::KernelCache& kernel_cache() { return *kcache_; }
  const durability::RecoveryReport& recovery_report() const { return recovery_; }
  std::vector<metrics::QueryMetrics> history() const { return metrics::Registry::Global().history(); }

 private:
  explicit Engine(EngineConfig cfg);
  Status Init();
  EngineConfig cfg_;
  hal::DevicePtr device_;
  std::unique_ptr<memory::MemoryManager> memory_;
  std::unique_ptr<storage::Catalog> catalog_;
  integration::CapabilityRegistry registry_;
  std::unique_ptr<exec::fused::KernelCache> kcache_;
  std::unique_ptr<exec::fused::JitCompiler> jit_;
  std::unique_ptr<ingest::WriteAheadLog> wal_;
  std::unique_ptr<ingest::DeltaStore> delta_;
  std::unique_ptr<ingest::Compactor> compactor_;
  std::unique_ptr<ingest::IngestService> ingest_;
  std::unique_ptr<planner::Planner> planner_;
  std::unique_ptr<exec::Executor> executor_;
  durability::RecoveryReport recovery_;
  uint64_t query_counter_ = 0;
};

}  // namespace aster
