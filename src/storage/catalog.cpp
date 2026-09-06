#include "aster/storage/catalog.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "aster/common/log.hpp"
#include "aster/memory/gds.hpp"

namespace aster::storage {

namespace fs = std::filesystem;

uint64_t SegmentList::total_rows() const {
  uint64_t n = 0;
  for (const auto& s : segments) n += s->num_rows;
  return n;
}

uint64_t SegmentList::total_bytes() const {
  uint64_t n = 0;
  for (const auto& s : segments) n += s->encoded_bytes;
  return n;
}

Catalog::Catalog(std::string root_dir) : root_(std::move(root_dir)) {}

std::string Catalog::ManifestPath(const std::string& table) const { return root_ + "/" + table + "/manifest.tsv"; }
std::string Catalog::SegmentPath(const std::string& table, SegmentId id) const {
  return root_ + "/" + table + "/seg_" + std::to_string(id) + ".aseg";
}

Status Catalog::Open() {
  ASTER_RETURN_NOT_OK(memory::EnsureDir(root_));
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root_, ec)) {
    if (!e.is_directory()) continue;
    std::string table = e.path().filename().string();
    Status s = LoadManifest(table);
    if (!s.ok()) ASTER_LOG(Warn, "table %s: %s", table.c_str(), s.ToString().c_str());
  }
  return Status::OK();
}

Status Catalog::LoadManifest(const std::string& table) {
  std::ifstream f(ManifestPath(table));
  if (!f) return Status::NotFound("manifest for " + table);
  auto list = std::make_shared<SegmentList>();
  list->table = table;
  TableInfo info;
  info.name = table;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream is(line);
    std::string kind;
    is >> kind;
    if (kind == "version") { is >> list->version; }
    else if (kind == "next_id") {
      SegmentId next; is >> next;
      SegmentId cur = next_segment_id_.load();
      while (next > cur && !next_segment_id_.compare_exchange_weak(cur, next)) {}
    }
    else if (kind == "field") {
      std::string name; int tid; uint32_t w, prec; int nullable;
      is >> name >> tid >> w >> prec >> nullable;
      info.schema.fields.push_back({name, DataType{static_cast<TypeId>(tid), w, prec}, nullable != 0});
    } else if (kind == "partition_key") { std::string k; is >> k; info.partition_keys.push_back(k); }
    else if (kind == "dimension") { int d; is >> d; info.dimension_hint = d != 0; }
    else if (kind == "segment") {
      SegmentId id; std::string path;
      is >> id >> path;
      auto meta = SegmentReader::ReadMeta(path);
      if (!meta.ok()) { ASTER_LOG(Warn, "skipping segment %s: %s", path.c_str(), meta.status().ToString().c_str()); continue; }
      list->segments.push_back(std::make_shared<SegmentMeta>(std::move(meta.value())));
      SegmentId next = id + 1;
      SegmentId cur = next_segment_id_.load();
      while (next > cur && !next_segment_id_.compare_exchange_weak(cur, next)) {}
    }
  }
  std::lock_guard<std::mutex> lk(mu_);
  tables_[table] = info;
  lists_[table] = list;
  return Status::OK();
}

Status Catalog::PersistManifest(const std::string& table, const SegmentList& list) const {
  std::ostringstream os;
  auto it = tables_.find(table);
  if (it == tables_.end()) return Status::NotFound("table " + table);
  os << "version " << list.version << "\n";
  os << "next_id " << next_segment_id_.load() << "\n";
  for (const auto& f : it->second.schema.fields)
    os << "field " << f.name << " " << int(f.type.id) << " " << f.type.width << " " << f.type.precision << " " << (f.nullable ? 1 : 0) << "\n";
  for (const auto& k : it->second.partition_keys) os << "partition_key " << k << "\n";
  os << "dimension " << (it->second.dimension_hint ? 1 : 0) << "\n";
  for (const auto& s : list.segments) os << "segment " << s->segment_id << " " << s->path << "\n";
  std::string body = os.str();
  std::string tmp = ManifestPath(table) + ".tmp";
  ASTER_RETURN_NOT_OK(memory::WriteFile(tmp, body.data(), body.size(), true));
  std::error_code ec;
  fs::rename(tmp, ManifestPath(table), ec);
  if (ec) return Status::IoError("rename manifest: " + ec.message());
  return Status::OK();
}

Status Catalog::CreateTable(const TableInfo& info) {
  std::lock_guard<std::mutex> lk(mu_);
  if (tables_.count(info.name)) return Status::Invalid("table exists: " + info.name);
  ASTER_RETURN_NOT_OK(memory::EnsureDir(root_ + "/" + info.name));
  tables_[info.name] = info;
  auto list = std::make_shared<SegmentList>();
  list->table = info.name;
  lists_[info.name] = list;
  return PersistManifest(info.name, *list);
}

Status Catalog::DropTable(const std::string& table) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!tables_.erase(table)) return Status::NotFound("table " + table);
  lists_.erase(table);
  std::error_code ec;
  fs::remove_all(root_ + "/" + table, ec);
  return Status::OK();
}

bool Catalog::HasTable(const std::string& table) const {
  std::lock_guard<std::mutex> lk(mu_);
  return tables_.count(table) > 0;
}

Result<TableInfo> Catalog::GetTable(const std::string& table) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = tables_.find(table);
  if (it == tables_.end()) return Status::NotFound("table " + table);
  return it->second;
}

std::vector<std::string> Catalog::Tables() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::string> out;
  for (const auto& [k, v] : tables_) out.push_back(k);
  return out;
}

Result<SegmentListPtr> Catalog::Snapshot(const std::string& table) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = lists_.find(table);
  if (it == lists_.end()) return Status::NotFound("table " + table);
  return it->second;
}

Status Catalog::SwapSegments(const std::string& table, std::vector<std::shared_ptr<SegmentMeta>> segments) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = lists_.find(table);
  if (it == lists_.end()) return Status::NotFound("table " + table);
  auto next = std::make_shared<SegmentList>();
  next->table = table;
  next->version = it->second->version + 1;
  next->segments = std::move(segments);
  ASTER_RETURN_NOT_OK(PersistManifest(table, *next));
  it->second = next;  // old snapshot stays alive for in flight queries
  return Status::OK();
}

Status Catalog::AppendSegment(const std::string& table, std::shared_ptr<SegmentMeta> seg) {
  ASTER_ASSIGN_OR_RETURN(auto snap, Snapshot(table));
  auto segs = snap->segments;
  segs.push_back(std::move(seg));
  return SwapSegments(table, std::move(segs));
}

Status Catalog::ReplaceSegments(const std::string& table, const std::vector<SegmentId>& remove,
                                std::vector<std::shared_ptr<SegmentMeta>> add) {
  ASTER_ASSIGN_OR_RETURN(auto snap, Snapshot(table));
  std::vector<std::shared_ptr<SegmentMeta>> segs;
  for (const auto& s : snap->segments) {
    bool drop = false;
    for (SegmentId r : remove) drop |= (r == s->segment_id);
    if (!drop) segs.push_back(s);
  }
  for (auto& a : add) segs.push_back(std::move(a));
  return SwapSegments(table, std::move(segs));
}

}  // namespace aster::storage
