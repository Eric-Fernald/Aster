#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/memory/page.hpp"
#include "aster/storage/bloom_filter.hpp"
#include "aster/storage/encoding.hpp"
#include "aster/storage/zone_map.hpp"

namespace aster::storage {

constexpr uint32_t kSegmentMagic = 0x47455341;  // "ASEG"
constexpr uint32_t kSegmentVersion = 1;
constexpr uint32_t kChunkAlignment = 4096;       // O_DIRECT / cuFile friendly

struct ColumnChunkMeta {
  std::string name;
  DataType type;
  Encoding encoding = Encoding::Plain;
  Encoding inner = Encoding::Plain;
  Codec codec = Codec::None;
  uint64_t offset = 0;
  uint32_t bytes = 0;
  uint32_t checksum = 0;
  uint32_t num_rows = 0;
  uint32_t null_count = 0;
  uint64_t dictionary_id = 0;
  ZoneMap zone;
  BloomFilter bloom;
};

struct SegmentMeta {
  SegmentId segment_id = 0;
  std::string path;
  std::string table;
  uint64_t num_rows = 0;
  uint64_t encoded_bytes = 0;
  uint32_t tile_rows = 0;
  std::vector<ColumnChunkMeta> columns;

  int ColumnIndex(const std::string& name) const;
  Schema schema() const;
  memory::PageDesc PageFor(ColumnId col) const;
  std::vector<memory::PageDesc> Pages() const;
  void Serialize(std::vector<uint8_t>& out) const;
  static Result<SegmentMeta> Deserialize(const uint8_t* p, size_t len);
};

struct SegmentWriteOptions {
  std::vector<Encoding> encodings;      // per column; empty = auto select
  Codec outer_codec = Codec::None;      // cold tier wrapping
  bool build_bloom = true;
  double bloom_fpp = 0.01;
  uint32_t tile_rows = 32 * 1024;
  std::vector<uint64_t> dictionary_ids;  // per column shared dictionary ids
};

// Writes one segment: contiguous encoded blobs, one per column, then a footer with all metadata.
class SegmentWriter {
 public:
  static Result<SegmentMeta> Write(const RecordBatch& batch, SegmentId id, const std::string& path,
                                   const std::string& table, const SegmentWriteOptions& opts);
};

class SegmentReader {
 public:
  static Result<SegmentMeta> ReadMeta(const std::string& path);
  static Result<std::vector<uint8_t>> ReadChunkBytes(const SegmentMeta& meta, ColumnId col);
  static Result<Column> ReadColumn(const SegmentMeta& meta, ColumnId col);
  static Result<RecordBatchPtr> ReadAll(const SegmentMeta& meta);
  static Result<RecordBatchPtr> ReadColumns(const SegmentMeta& meta, const std::vector<ColumnId>& cols);
  static Status VerifyChunk(const ColumnChunkMeta& c, const uint8_t* data, size_t len);
};

}  // namespace aster::storage
