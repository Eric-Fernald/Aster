#include "aster/exec/operators/scan.hpp"

#include <algorithm>

#include "aster/common/log.hpp"
#include "aster/metrics/metrics.hpp"
#include "aster/storage/segment.hpp"

namespace aster::exec {

SegmentScanSource::SegmentScanSource(const planner::SegmentScan& scan, const Schema& output, const std::vector<int>& projection,
                                     const Schema& table_schema, ExecContext& ctx, std::vector<RecordBatchPtr> delta_batches)
    : scan_(scan), output_(output), projection_(projection), table_schema_(table_schema), ctx_(ctx), delta_(std::move(delta_batches)) {}

uint64_t SegmentScanSource::estimated_tiles() const {
  uint64_t rows = scan_.rows;
  return (rows + ctx_.tile_rows - 1) / ctx_.tile_rows;
}

Status SegmentScanSource::LoadSegment(size_t index) {
  const auto& seg = *scan_.segments[index];
  auto batch = std::make_shared<RecordBatch>();
  batch->schema = output_;
  cur_pins_.clear();
  cur_encodings_.clear();
  uint64_t bytes = 0;
  for (size_t oi = 0; oi < projection_.size(); ++oi) {
    const std::string& name = table_schema_.fields.at(projection_[oi]).name;
    int ci = seg.ColumnIndex(name);
    if (ci < 0) return Status::NotFound("column " + name + " in segment " + std::to_string(seg.segment_id));
    const storage::ColumnChunkMeta& cm = seg.columns[ci];
    Column col;
    if (ctx_.memory) {
      // Page through the memory manager: encoded bytes land in the top tier and stay pinned for this segment.
      ASTER_ASSIGN_OR_RETURN(auto handle, ctx_.memory->Acquire({seg.segment_id, static_cast<ColumnId>(ci)}, ctx_.stream));
      const uint8_t* data = static_cast<const uint8_t*>(handle.data());
      if (handle.space() == MemorySpace::Device) {
        ASTER_ASSIGN_OR_RETURN(auto host, ctx_.memory->device().ToHost(Buffer(const_cast<void*>(handle.data()), handle.bytes(), MemorySpace::Device), ctx_.stream));
        ASTER_RETURN_NOT_OK(ctx_.memory->device().Synchronize(ctx_.stream));
        ASTER_ASSIGN_OR_RETURN(col, storage::DecodeChunk(host->data(), host->size()));
      } else {
        ASTER_RETURN_NOT_OK(storage::SegmentReader::VerifyChunk(cm, data, handle.bytes()));
        ASTER_ASSIGN_OR_RETURN(col, storage::DecodeChunk(data, handle.bytes()));
      }
      cur_pins_.push_back(std::move(handle));
    } else {
      ASTER_ASSIGN_OR_RETURN(col, storage::SegmentReader::ReadColumn(seg, static_cast<ColumnId>(ci)));
    }
    col.dictionary_id = cm.dictionary_id;
    if (!ctx_.compressed_execution && col.is_dictionary_encoded()) col = col.DecodeDictionary();
    bytes += cm.bytes;
    cur_encodings_.push_back(cm.encoding);
    batch->columns.push_back(std::move(col));
  }
  cur_ = batch;
  cur_offset_ = 0;
  cur_segment_ = seg.segment_id;
  if (ctx_.on_scan) ctx_.on_scan(seg.num_rows, bytes);
  metrics::Registry::Global().counter("rows_scanned").Add(seg.num_rows);
  metrics::Registry::Global().counter("bytes_scanned_hbm").Add(bytes);
  return Status::OK();
}

Result<bool> SegmentScanSource::Next(Tile* out) {
  for (;;) {
    if (cur_ && cur_offset_ < cur_->num_rows()) {
      int64_t n = std::min<int64_t>(ctx_.tile_rows, cur_->num_rows() - cur_offset_);
      out->batch = (cur_offset_ == 0 && n == cur_->num_rows()) ? cur_ : SliceBatch(*cur_, cur_offset_, n);
      out->selection = SelectionVector::All();
      out->tile_index = tile_counter_++;
      out->segment_id = cur_segment_;
      out->run_lengths.clear();
      cur_offset_ += n;
      return true;
    }
    if (seg_index_ < scan_.segments.size()) {
      ASTER_RETURN_NOT_OK(LoadSegment(seg_index_++));
      continue;
    }
    if (delta_index_ < delta_.size()) {
      // Delta store rows are unioned with base segments: same schema, projected to the read's columns.
      const RecordBatch& d = *delta_[delta_index_++];
      auto b = std::make_shared<RecordBatch>();
      b->schema = output_;
      for (int p : projection_) {
        if (p >= static_cast<int>(d.columns.size())) return Status::Invalid("delta batch narrower than table");
        b->columns.push_back(d.columns[p]);
      }
      cur_ = b;
      cur_offset_ = 0;
      cur_segment_ = 0;
      cur_pins_.clear();
      if (ctx_.on_scan) ctx_.on_scan(b->num_rows(), b->nbytes());
      continue;
    }
    cur_.reset();
    cur_pins_.clear();
    return false;
  }
}

BatchSource::BatchSource(Schema schema, std::vector<RecordBatchPtr> batches, uint32_t tile_rows)
    : schema_(std::move(schema)), batches_(std::move(batches)), tile_rows_(tile_rows ? tile_rows : 1) {}

uint64_t BatchSource::estimated_tiles() const {
  uint64_t n = 0;
  for (const auto& b : batches_) n += (b->num_rows() + tile_rows_ - 1) / tile_rows_;
  return n;
}

Result<bool> BatchSource::Next(Tile* out) {
  while (index_ < batches_.size()) {
    const RecordBatch& b = *batches_[index_];
    if (offset_ >= b.num_rows()) { ++index_; offset_ = 0; continue; }
    int64_t n = std::min<int64_t>(tile_rows_, b.num_rows() - offset_);
    out->batch = (offset_ == 0 && n == b.num_rows()) ? batches_[index_] : SliceBatch(b, offset_, n);
    out->selection = SelectionVector::All();
    out->tile_index = tile_counter_++;
    out->segment_id = 0;
    out->run_lengths.clear();
    offset_ += n;
    return true;
  }
  return false;
}

}  // namespace aster::exec
