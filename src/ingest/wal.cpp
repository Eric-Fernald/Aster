#include "aster/ingest/wal.hpp"

#include <cstring>
#include <filesystem>

#include "aster/common/hash.hpp"
#include "aster/common/log.hpp"
#include "aster/durability/fault_injector.hpp"
#include "aster/memory/gds.hpp"

namespace aster::ingest {

namespace {
constexpr uint32_t kRecordMagic = 0x4C415741;  // "AWAL"
struct RecordHeader {
  uint32_t magic;
  uint32_t length;   // payload bytes
  uint64_t lsn;
  uint32_t crc;      // over payload
  uint32_t reserved;
};
template <typename T> void Put(std::vector<uint8_t>& o, const T& v) { const uint8_t* p = reinterpret_cast<const uint8_t*>(&v); o.insert(o.end(), p, p + sizeof v); }
void PutBytes(std::vector<uint8_t>& o, const void* p, size_t n) { o.insert(o.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); }
template <typename T> T Get(const uint8_t*& p) { T v; std::memcpy(&v, p, sizeof v); p += sizeof v; return v; }
}  // namespace

std::vector<uint8_t> WriteAheadLog::SerializeBatch(const std::string& table, const RecordBatch& batch) {
  std::vector<uint8_t> out;
  Put<uint32_t>(out, static_cast<uint32_t>(table.size()));
  PutBytes(out, table.data(), table.size());
  Put<uint32_t>(out, static_cast<uint32_t>(batch.columns.size()));
  Put<int64_t>(out, batch.num_rows());
  for (size_t i = 0; i < batch.columns.size(); ++i) {
    Column c = batch.columns[i].is_dictionary_encoded() ? batch.columns[i].DecodeDictionary() : batch.columns[i];
    const Field& f = batch.schema.fields[i];
    Put<uint32_t>(out, static_cast<uint32_t>(f.name.size()));
    PutBytes(out, f.name.data(), f.name.size());
    Put<uint8_t>(out, static_cast<uint8_t>(c.type.id));
    Put<uint32_t>(out, c.type.width);
    Put<uint32_t>(out, c.type.precision);
    Put<int64_t>(out, c.null_count);
    uint32_t vb = c.validity ? static_cast<uint32_t>(c.validity->size()) : 0;
    Put<uint32_t>(out, vb);
    if (vb) PutBytes(out, c.validity->data(), vb);
    uint32_t ob = c.offsets ? static_cast<uint32_t>(c.offsets->size()) : 0;
    Put<uint32_t>(out, ob);
    if (ob) PutBytes(out, c.offsets->data(), ob);
    uint32_t nb = c.values ? static_cast<uint32_t>(IsFixedWidth(c.type.id) ? TypeByteWidth(c.type) * c.length : c.offsets->as<int32_t>()[c.length]) : 0;
    Put<uint32_t>(out, nb);
    if (nb) PutBytes(out, c.values->data(), nb);
  }
  return out;
}

Result<WriteAheadLog::Record> WriteAheadLog::DeserializeBatch(const uint8_t* data, size_t len) {
  const uint8_t* p = data;
  const uint8_t* end = data + len;
  Record r;
  uint32_t tn = Get<uint32_t>(p);
  if (p + tn > end) return Status::Corrupt("wal table name");
  r.table.assign(reinterpret_cast<const char*>(p), tn); p += tn;
  uint32_t ncols = Get<uint32_t>(p);
  int64_t rows = Get<int64_t>(p);
  r.batch = std::make_shared<RecordBatch>();
  for (uint32_t i = 0; i < ncols; ++i) {
    if (p >= end) return Status::Corrupt("wal column truncated");
    uint32_t nn = Get<uint32_t>(p);
    std::string name(reinterpret_cast<const char*>(p), nn); p += nn;
    Column c;
    c.type.id = static_cast<TypeId>(Get<uint8_t>(p));
    c.type.width = Get<uint32_t>(p);
    c.type.precision = Get<uint32_t>(p);
    c.length = rows;
    c.null_count = Get<int64_t>(p);
    uint32_t vb = Get<uint32_t>(p);
    if (vb) { c.validity = Buffer::CopyOf(p, vb); p += vb; }
    uint32_t ob = Get<uint32_t>(p);
    if (ob) { c.offsets = Buffer::CopyOf(p, ob); p += ob; }
    uint32_t nb = Get<uint32_t>(p);
    if (p + nb > end) return Status::Corrupt("wal values truncated");
    c.values = Buffer::CopyOf(p, nb); p += nb;
    r.batch->schema.fields.push_back({name, c.type, c.null_count > 0});
    r.batch->columns.push_back(std::move(c));
  }
  return r;
}

WriteAheadLog::WriteAheadLog(std::string dir) : dir_(std::move(dir)), path_(dir_ + "/aster.wal") {}

Status WriteAheadLog::Open() {
  ASTER_RETURN_NOT_OK(memory::EnsureDir(dir_));
  if (!memory::FileExists(path_)) return memory::WriteFile(path_, "", 0, true);
  auto sz = memory::FileSize(path_);
  bytes_ = sz.ok() ? sz.value() : 0;
  // Scan for the last valid LSN so new appends continue the sequence.
  return Replay([&](const Record&) { return Status::OK(); });
}

Result<uint64_t> WriteAheadLog::Append(const std::string& table, const RecordBatch& batch, bool fsync) {
  std::vector<uint8_t> payload = SerializeBatch(table, batch);
  std::lock_guard<std::mutex> lk(mu_);
  RecordHeader h{kRecordMagic, static_cast<uint32_t>(payload.size()), last_lsn_ + 1, Crc32c(payload.data(), payload.size()), 0};
  std::vector<uint8_t> rec;
  PutBytes(rec, &h, sizeof h);
  ASTER_FAULT_POINT("wal.before_write");
  rec.insert(rec.end(), payload.begin(), payload.end());
  ASTER_RETURN_NOT_OK(memory::AppendFile(path_, rec.data(), rec.size(), fsync));
  ASTER_FAULT_POINT("wal.after_write");
  last_lsn_ = h.lsn;
  bytes_ += rec.size();
  return h.lsn;
}

Status WriteAheadLog::Replay(const std::function<Status(const Record&)>& fn) {
  ASTER_ASSIGN_OR_RETURN(auto bytes, memory::ReadWholeFile(path_));
  const uint8_t* p = bytes.data();
  const uint8_t* end = p + bytes.size();
  uint64_t valid_bytes = 0;
  while (p + sizeof(RecordHeader) <= end) {
    RecordHeader h;
    std::memcpy(&h, p, sizeof h);
    if (h.magic != kRecordMagic || p + sizeof h + h.length > end) break;  // torn tail
    const uint8_t* payload = p + sizeof h;
    if (Crc32c(payload, h.length) != h.crc) { ASTER_LOG(Warn, "wal record %llu failed checksum, truncating", (unsigned long long)h.lsn); break; }
    auto rec = DeserializeBatch(payload, h.length);
    if (!rec.ok()) return rec.status();
    rec.value().lsn = h.lsn;
    ASTER_RETURN_NOT_OK(fn(rec.value()));
    last_lsn_ = std::max(last_lsn_, h.lsn);
    p += sizeof h + h.length;
    valid_bytes = static_cast<uint64_t>(p - bytes.data());
  }
  if (valid_bytes != bytes.size()) {
    ASTER_LOG(Warn, "wal: dropping %zu torn bytes at tail", bytes.size() - valid_bytes);
    ASTER_RETURN_NOT_OK(memory::WriteFile(path_, bytes.data(), valid_bytes, true));
  }
  bytes_ = valid_bytes;
  return Status::OK();
}

Status WriteAheadLog::Truncate(uint64_t up_to_lsn) {
  std::lock_guard<std::mutex> lk(mu_);
  ASTER_ASSIGN_OR_RETURN(auto bytes, memory::ReadWholeFile(path_));
  std::vector<uint8_t> keep;
  const uint8_t* p = bytes.data();
  const uint8_t* end = p + bytes.size();
  while (p + sizeof(RecordHeader) <= end) {
    RecordHeader h;
    std::memcpy(&h, p, sizeof h);
    if (h.magic != kRecordMagic || p + sizeof h + h.length > end) break;
    if (h.lsn > up_to_lsn) keep.insert(keep.end(), p, p + sizeof h + h.length);
    p += sizeof h + h.length;
  }
  std::string tmp = path_ + ".tmp";
  ASTER_RETURN_NOT_OK(memory::WriteFile(tmp, keep.data(), keep.size(), true));
  std::error_code ec;
  std::filesystem::rename(tmp, path_, ec);
  if (ec) return Status::IoError("wal rename: " + ec.message());
  bytes_ = keep.size();
  return Status::OK();
}

}  // namespace aster::ingest
