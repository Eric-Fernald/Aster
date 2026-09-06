#include "aster/storage/segment.hpp"

#include <fstream>

#include "aster/common/hash.hpp"
#include "aster/common/log.hpp"
#include "aster/memory/gds.hpp"
#include "aster/storage/encoding_selector.hpp"
#include "internal.hpp"

namespace aster::storage {

using namespace internal;

namespace {
struct FileHeader {
  uint32_t magic = kSegmentMagic;
  uint32_t version = kSegmentVersion;
  uint64_t segment_id = 0;
  uint64_t num_rows = 0;
  uint32_t num_columns = 0;
  uint32_t tile_rows = 0;
  uint64_t footer_offset = 0;
  uint32_t footer_bytes = 0;
  uint32_t footer_crc = 0;
};
static_assert(sizeof(FileHeader) == 48);

void PutStr(std::vector<uint8_t>& out, const std::string& s) {
  PutVal<uint32_t>(out, static_cast<uint32_t>(s.size()));
  Put(out, s.data(), s.size());
}
std::string GetStr(const uint8_t*& p) {
  uint32_t n = GetVal<uint32_t>(p);
  std::string s(reinterpret_cast<const char*>(p), n);
  p += n;
  return s;
}
size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }
}  // namespace

int SegmentMeta::ColumnIndex(const std::string& name) const {
  for (size_t i = 0; i < columns.size(); ++i)
    if (columns[i].name == name) return static_cast<int>(i);
  return -1;
}

Schema SegmentMeta::schema() const {
  Schema s;
  for (const auto& c : columns) s.fields.push_back({c.name, c.type, c.null_count > 0});
  return s;
}

memory::PageDesc SegmentMeta::PageFor(ColumnId col) const {
  const auto& c = columns.at(col);
  memory::PageDesc d;
  d.segment_id = segment_id;
  d.column_id = col;
  d.encoded_bytes = c.bytes;
  d.encoding = static_cast<uint8_t>(c.encoding);
  d.tier = static_cast<uint8_t>(memory::Tier::Nvme);
  d.locator = {path, c.offset, c.bytes, c.checksum};
  return d;
}

std::vector<memory::PageDesc> SegmentMeta::Pages() const {
  std::vector<memory::PageDesc> out;
  for (ColumnId i = 0; i < columns.size(); ++i) out.push_back(PageFor(i));
  return out;
}

void SegmentMeta::Serialize(std::vector<uint8_t>& out) const {
  PutVal<uint64_t>(out, segment_id);
  PutStr(out, table);
  PutVal<uint64_t>(out, num_rows);
  PutVal<uint64_t>(out, encoded_bytes);
  PutVal<uint32_t>(out, tile_rows);
  PutVal<uint32_t>(out, static_cast<uint32_t>(columns.size()));
  for (const auto& c : columns) {
    PutStr(out, c.name);
    PutVal<uint8_t>(out, static_cast<uint8_t>(c.type.id));
    PutVal<uint32_t>(out, c.type.width);
    PutVal<uint32_t>(out, c.type.precision);
    PutVal<uint8_t>(out, static_cast<uint8_t>(c.encoding));
    PutVal<uint8_t>(out, static_cast<uint8_t>(c.inner));
    PutVal<uint8_t>(out, static_cast<uint8_t>(c.codec));
    PutVal<uint64_t>(out, c.offset);
    PutVal<uint32_t>(out, c.bytes);
    PutVal<uint32_t>(out, c.checksum);
    PutVal<uint32_t>(out, c.num_rows);
    PutVal<uint32_t>(out, c.null_count);
    PutVal<uint64_t>(out, c.dictionary_id);
    c.zone.Serialize(out);
    PutVal<uint8_t>(out, !c.bloom.empty());
    if (!c.bloom.empty()) c.bloom.Serialize(out);
  }
}

Result<SegmentMeta> SegmentMeta::Deserialize(const uint8_t* p, size_t len) {
  const uint8_t* end = p + len;
  SegmentMeta m;
  m.segment_id = GetVal<uint64_t>(p);
  m.table = GetStr(p);
  m.num_rows = GetVal<uint64_t>(p);
  m.encoded_bytes = GetVal<uint64_t>(p);
  m.tile_rows = GetVal<uint32_t>(p);
  uint32_t n = GetVal<uint32_t>(p);
  for (uint32_t i = 0; i < n; ++i) {
    if (p >= end) return Status::Corrupt("segment footer truncated");
    ColumnChunkMeta c;
    c.name = GetStr(p);
    c.type.id = static_cast<TypeId>(GetVal<uint8_t>(p));
    c.type.width = GetVal<uint32_t>(p);
    c.type.precision = GetVal<uint32_t>(p);
    c.encoding = static_cast<Encoding>(GetVal<uint8_t>(p));
    c.inner = static_cast<Encoding>(GetVal<uint8_t>(p));
    c.codec = static_cast<Codec>(GetVal<uint8_t>(p));
    c.offset = GetVal<uint64_t>(p);
    c.bytes = GetVal<uint32_t>(p);
    c.checksum = GetVal<uint32_t>(p);
    c.num_rows = GetVal<uint32_t>(p);
    c.null_count = GetVal<uint32_t>(p);
    c.dictionary_id = GetVal<uint64_t>(p);
    c.zone = ZoneMap::Deserialize(p, end);
    if (GetVal<uint8_t>(p)) c.bloom = BloomFilter::Deserialize(p, end);
    m.columns.push_back(std::move(c));
  }
  return m;
}

Result<SegmentMeta> SegmentWriter::Write(const RecordBatch& batch, SegmentId id, const std::string& path,
                                         const std::string& table, const SegmentWriteOptions& opts) {
  SegmentMeta meta;
  meta.segment_id = id;
  meta.path = path;
  meta.table = table;
  meta.num_rows = batch.num_rows();
  meta.tile_rows = opts.tile_rows;

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return Status::IoError("cannot create " + path);
  FileHeader hdr;
  hdr.segment_id = id;
  hdr.num_rows = meta.num_rows;
  hdr.num_columns = static_cast<uint32_t>(batch.columns.size());
  hdr.tile_rows = opts.tile_rows;
  f.write(reinterpret_cast<const char*>(&hdr), sizeof hdr);
  uint64_t pos = sizeof hdr;

  for (size_t ci = 0; ci < batch.columns.size(); ++ci) {
    const Column& col = batch.columns[ci];
    EncodeOptions eo;
    eo.encoding = ci < opts.encodings.size() ? opts.encodings[ci] : SelectEncoding(col);
    eo.outer_codec = opts.outer_codec;
    eo.dictionary_id = ci < opts.dictionary_ids.size() ? opts.dictionary_ids[ci] : 0;
    auto enc = EncodeColumn(col, eo);
    if (!enc.ok()) {
      eo.encoding = Encoding::Plain;
      ASTER_ASSIGN_OR_RETURN(enc, EncodeColumn(col, eo));
    }
    const EncodedChunk& chunk = enc.value();
    uint64_t aligned = AlignUp(pos, kChunkAlignment);
    if (aligned > pos) { std::vector<char> pad(aligned - pos, 0); f.write(pad.data(), pad.size()); pos = aligned; }

    ColumnChunkMeta cm;
    cm.name = batch.schema.fields[ci].name;
    cm.type = col.type;
    cm.encoding = chunk.encoding;
    cm.inner = chunk.inner;
    cm.codec = chunk.codec;
    cm.offset = pos;
    cm.bytes = static_cast<uint32_t>(chunk.data.size());
    cm.checksum = Crc32c(chunk.data.data(), chunk.data.size());
    cm.num_rows = chunk.num_rows;
    cm.null_count = chunk.null_count;
    cm.dictionary_id = chunk.dictionary_id;
    cm.zone = chunk.zone;
    if (opts.build_bloom && (col.type.id == TypeId::String || IsIntegerLike(col.type.id)))
      cm.bloom = BloomFilter::Build(col, opts.bloom_fpp);
    f.write(reinterpret_cast<const char*>(chunk.data.data()), chunk.data.size());
    pos += chunk.data.size();
    meta.encoded_bytes += chunk.data.size();
    meta.columns.push_back(std::move(cm));
  }

  std::vector<uint8_t> footer;
  meta.Serialize(footer);
  hdr.footer_offset = pos;
  hdr.footer_bytes = static_cast<uint32_t>(footer.size());
  hdr.footer_crc = Crc32c(footer.data(), footer.size());
  f.write(reinterpret_cast<const char*>(footer.data()), footer.size());
  f.write(reinterpret_cast<const char*>(&hdr.footer_offset), sizeof(uint64_t));
  f.write(reinterpret_cast<const char*>(&hdr.magic), sizeof(uint32_t));
  f.seekp(0);
  f.write(reinterpret_cast<const char*>(&hdr), sizeof hdr);
  f.flush();
  if (!f) return Status::IoError("write failed " + path);
  return meta;
}

Result<SegmentMeta> SegmentReader::ReadMeta(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Status::NotFound(path);
  FileHeader hdr;
  f.read(reinterpret_cast<char*>(&hdr), sizeof hdr);
  if (!f || hdr.magic != kSegmentMagic) return Status::Corrupt("bad segment header " + path);
  if (hdr.version != kSegmentVersion) return Status::NotSupported("segment version " + std::to_string(hdr.version));
  std::vector<uint8_t> footer(hdr.footer_bytes);
  f.seekg(static_cast<std::streamoff>(hdr.footer_offset));
  f.read(reinterpret_cast<char*>(footer.data()), footer.size());
  if (!f) return Status::Corrupt("footer truncated " + path);
  if (Crc32c(footer.data(), footer.size()) != hdr.footer_crc) return Status::Corrupt("footer checksum " + path);
  ASTER_ASSIGN_OR_RETURN(SegmentMeta m, SegmentMeta::Deserialize(footer.data(), footer.size()));
  m.path = path;
  return m;
}

Status SegmentReader::VerifyChunk(const ColumnChunkMeta& c, const uint8_t* data, size_t len) {
  if (len != c.bytes) return Status::Corrupt("chunk size mismatch for " + c.name);
  if (Crc32c(data, len) != c.checksum) return Status::Corrupt("chunk checksum mismatch for " + c.name);
  return Status::OK();
}

Result<std::vector<uint8_t>> SegmentReader::ReadChunkBytes(const SegmentMeta& meta, ColumnId col) {
  const auto& c = meta.columns.at(col);
  std::vector<uint8_t> buf(c.bytes);
  memory::PosixFileReader r(nullptr);
  ASTER_RETURN_NOT_OK(r.ReadToHost(meta.path, c.offset, c.bytes, buf.data()));
  ASTER_RETURN_NOT_OK(VerifyChunk(c, buf.data(), buf.size()));
  return buf;
}

Result<Column> SegmentReader::ReadColumn(const SegmentMeta& meta, ColumnId col) {
  ASTER_ASSIGN_OR_RETURN(auto bytes, ReadChunkBytes(meta, col));
  ASTER_ASSIGN_OR_RETURN(Column c, DecodeChunk(bytes.data(), bytes.size()));
  c.dictionary_id = meta.columns[col].dictionary_id;
  return c;
}

Result<RecordBatchPtr> SegmentReader::ReadColumns(const SegmentMeta& meta, const std::vector<ColumnId>& cols) {
  auto out = std::make_shared<RecordBatch>();
  for (ColumnId c : cols) {
    ASTER_ASSIGN_OR_RETURN(Column col, ReadColumn(meta, c));
    out->schema.fields.push_back({meta.columns[c].name, meta.columns[c].type, meta.columns[c].null_count > 0});
    out->columns.push_back(std::move(col));
  }
  return out;
}

Result<RecordBatchPtr> SegmentReader::ReadAll(const SegmentMeta& meta) {
  std::vector<ColumnId> cols;
  for (ColumnId i = 0; i < meta.columns.size(); ++i) cols.push_back(i);
  return ReadColumns(meta, cols);
}

}  // namespace aster::storage
