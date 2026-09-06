#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/memory/memory_manager.hpp"

namespace aster::exec {

using QueryId = uint64_t;

// A tile is a fixed number of rows that one thread block (or one CPU worker) carries from scan
// through the next pipeline breaker. The selection vector implements late materialization.
struct Tile {
  RecordBatchPtr batch;
  SelectionVector selection;
  uint32_t tile_index = 0;
  SegmentId segment_id = 0;
  std::vector<memory::PageHandle> pins;  // keeps encoded pages resident while the tile is live
  std::vector<uint32_t> run_lengths;     // non empty when the batch carries run level rows
  int64_t num_rows() const { return batch ? selection.count(batch->num_rows()) : 0; }
  bool has_runs() const { return !run_lengths.empty(); }
  RecordBatchPtr Materialize() const { return TakeBatch(*batch, selection); }
};

class TileSource {
 public:
  virtual ~TileSource() = default;
  virtual Result<bool> Next(Tile* out) = 0;  // false when exhausted
  virtual uint64_t estimated_tiles() const = 0;
  virtual const Schema& schema() const = 0;
};

struct ExecContext {
  QueryId query_id = 0;
  uint32_t tile_rows = 32 * 1024;
  memory::MemoryManager* memory = nullptr;
  hal::Device* device = nullptr;
  hal::Stream stream;
  bool compressed_execution = true;
  uint32_t worker_threads = 0;  // 0 = hardware concurrency
  std::function<void(uint64_t rows, uint64_t bytes)> on_scan;
};

}  // namespace aster::exec
