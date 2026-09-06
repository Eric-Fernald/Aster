#pragma once
#include <deque>
#include <memory>
#include <vector>

#include "aster/exec/tile.hpp"
#include "aster/planner/physical_plan.hpp"
#include "aster/storage/encoding.hpp"

namespace aster::exec {

// Streams tiles out of native segments. Each column chunk is one page: acquired from the memory
// manager, decoded once per segment (dictionary columns stay as codes), then sliced into tiles.
class SegmentScanSource final : public TileSource {
 public:
  SegmentScanSource(const planner::SegmentScan& scan, const Schema& output, const std::vector<int>& projection,
                    const Schema& table_schema, ExecContext& ctx, std::vector<RecordBatchPtr> delta_batches = {});
  Result<bool> Next(Tile* out) override;
  uint64_t estimated_tiles() const override;
  const Schema& schema() const override { return output_; }
  // Encoded domain descriptions for the current segment, keyed by output column index.
  const std::vector<storage::Encoding>& current_encodings() const { return cur_encodings_; }

 private:
  Status LoadSegment(size_t index);
  const planner::SegmentScan& scan_;
  Schema output_;
  std::vector<int> projection_;
  Schema table_schema_;
  ExecContext& ctx_;
  std::vector<RecordBatchPtr> delta_;
  size_t seg_index_ = 0;
  size_t delta_index_ = 0;
  RecordBatchPtr cur_;
  std::vector<memory::PageHandle> cur_pins_;
  std::vector<storage::Encoding> cur_encodings_;
  int64_t cur_offset_ = 0;
  uint32_t tile_counter_ = 0;
  SegmentId cur_segment_ = 0;
};

class BatchSource final : public TileSource {
 public:
  BatchSource(Schema schema, std::vector<RecordBatchPtr> batches, uint32_t tile_rows);
  Result<bool> Next(Tile* out) override;
  uint64_t estimated_tiles() const override;
  const Schema& schema() const override { return schema_; }

 private:
  Schema schema_;
  std::vector<RecordBatchPtr> batches_;
  uint32_t tile_rows_;
  size_t index_ = 0;
  int64_t offset_ = 0;
  uint32_t tile_counter_ = 0;
};

}  // namespace aster::exec
