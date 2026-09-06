#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace aster {

enum class HardwareMode { Discrete, Coherent, CpuOnly };

struct EngineConfig {
  HardwareMode mode = HardwareMode::CpuOnly;
  std::vector<int> device_ids = {0};
  uint32_t tile_rows = 32 * 1024;
  size_t vram_budget_bytes = 0;  // 0 = probe device
  size_t host_pinned_budget_bytes = size_t(64) << 30;
  size_t segment_target_bytes = size_t(128) << 20;
  std::string data_dir = "/var/tmp/aster/data";
  std::string nvme_spill_dir = "/var/tmp/aster/spill";
  std::string wal_dir = "/var/tmp/aster/wal";
  std::string kernel_cache_dir = "/var/tmp/aster/kernels";
  std::string bandwidth_cache_path = "/var/tmp/aster/bandwidth.tsv";
  std::string capability_registry_path;
  uint32_t prefetch_depth = 4;
  bool enable_fusion = true;
  bool enable_compressed_execution = true;
  bool prefer_cudf_operators = true;
  bool enable_gds = true;
  bool log_plans = true;
  size_t delta_store_max_bytes = size_t(1) << 30;
  double gpu_hourly_cost_usd = 3.0;
  double cpu_hourly_cost_usd = 1.0;
};

const char* HardwareModeName(HardwareMode m);

}  // namespace aster
