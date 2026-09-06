#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aster/integration/plan_ir.hpp"
#include "aster/memory/page.hpp"
#include "aster/storage/segment.hpp"

namespace aster::planner {

enum class PlacementMode { Resident, StreamHost, StreamNvme, Cpu };
const char* PlacementModeName(PlacementMode m);

struct SegmentScan {
  int read_node_id = 0;
  std::string table;
  std::vector<std::shared_ptr<storage::SegmentMeta>> segments;  // survivors of pruning
  std::vector<ColumnId> columns;                                // physical columns actually referenced
  uint64_t rows = 0;
  uint64_t bytes = 0;
  uint64_t pruned_segments = 0;
  uint64_t pruned_rows = 0;
  bool include_delta = true;
};

// One fused unit of work: a chain of operators from a source to the next pipeline breaker.
struct Pipeline {
  int id = 0;
  std::vector<plan::RelPtr> ops;   // source first, breaker (or output sink) last
  std::vector<int> depends_on;     // pipelines that must complete first (join builds, sort inputs)
  std::string shape_hash;          // operator sequence + types + encodings, keyed for the kernel cache
  PlacementMode placement = PlacementMode::Resident;
  bool fusable = false;
  bool cpu = false;
  uint64_t est_rows_in = 0;
  uint64_t est_rows_out = 0;
  uint64_t est_bytes_hbm = 0;
  uint64_t est_bytes_host_to_hbm = 0;
  uint64_t est_bytes_nvme_to_hbm = 0;
  double est_seconds = 0;
  std::vector<memory::PageKey> pages;
  int scan_index = -1;             // index into PhysicalPlan::scans when the source is a Read
  const plan::Rel* source() const { return ops.empty() ? nullptr : ops.front().get(); }
  const plan::Rel* sink() const { return ops.empty() ? nullptr : ops.back().get(); }
  std::string ToString() const;
};

struct PhysicalPlan {
  plan::RelPtr root;
  std::vector<Pipeline> pipelines;  // topological order
  std::vector<SegmentScan> scans;
  std::vector<memory::PageKey> prefetch_order;
  uint64_t est_bytes_hbm = 0;
  uint64_t est_bytes_host_to_hbm = 0;
  uint64_t est_bytes_nvme_to_hbm = 0;
  uint64_t est_rows_scanned = 0;
  uint64_t segments_pruned = 0;
  uint64_t segments_scanned = 0;
  double est_seconds = 0;
  uint32_t kernel_cache_hits = 0;
  uint32_t kernel_cache_misses = 0;
  uint64_t est_bytes_moved() const { return est_bytes_host_to_hbm + est_bytes_nvme_to_hbm; }
  std::string ToString() const;
};

}  // namespace aster::planner
