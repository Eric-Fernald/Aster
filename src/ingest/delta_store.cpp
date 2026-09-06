#include "aster/ingest/delta_store.hpp"

#include <cstring>

namespace aster::ingest {

DeltaStore::DeltaStore(hal::DevicePtr dev, size_t max_bytes) : dev_(std::move(dev)), max_bytes_(max_bytes) {}

RecordBatchPtr DeltaStore::ToPinned(const RecordBatch& b) const {
  // Batches move into pinned host memory so a query can ship them to the GPU without a staging copy.
  auto out = std::make_shared<RecordBatch>();
  out->schema = b.schema;
  auto pin = [&](const std::shared_ptr<Buffer>& src) -> std::shared_ptr<Buffer> {
    if (!src) return nullptr;
    if (src->space() == MemorySpace::HostPinned || !dev_) return src;
    auto p = dev_->MakePinnedBuffer(src->size());
    if (!p) return src;
    std::memcpy(p->data(), src->data(), src->size());
    return p;
  };
  for (const auto& c : b.columns) {
    Column pc = c;
    pc.validity = pin(c.validity);
    pc.offsets = pin(c.offsets);
    pc.values = pin(c.values);
    out->columns.push_back(std::move(pc));
  }
  return out;
}

Status DeltaStore::Append(const std::string& table, RecordBatchPtr batch, uint64_t lsn) {
  if (!batch) return Status::Invalid("null batch");
  RecordBatchPtr pinned = ToPinned(*batch);
  std::lock_guard<std::mutex> lk(mu_);
  bytes_ += pinned->nbytes();
  tables_[table].push_back({std::move(pinned), lsn});
  return Status::OK();
}

DeltaSnapshot DeltaStore::Snapshot(const std::string& table) const {
  std::lock_guard<std::mutex> lk(mu_);
  DeltaSnapshot s;
  auto it = tables_.find(table);
  if (it == tables_.end()) return s;
  for (const auto& e : it->second) {
    s.batches.push_back(e.batch);
    s.rows += e.batch->num_rows();
    s.bytes += e.batch->nbytes();
    s.last_lsn = std::max(s.last_lsn, e.lsn);
  }
  return s;
}

void DeltaStore::Release(const std::string& table, uint64_t up_to_lsn) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = tables_.find(table);
  if (it == tables_.end()) return;
  auto& v = it->second;
  std::vector<Entry> keep;
  for (auto& e : v) {
    if (e.lsn <= up_to_lsn) bytes_ -= std::min(bytes_, e.batch->nbytes());
    else keep.push_back(std::move(e));
  }
  v = std::move(keep);
}

uint64_t DeltaStore::Rows(const std::string& table) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = tables_.find(table);
  if (it == tables_.end()) return 0;
  uint64_t n = 0;
  for (const auto& e : it->second) n += e.batch->num_rows();
  return n;
}

size_t DeltaStore::bytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return bytes_;
}

std::vector<std::string> DeltaStore::Tables() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::string> out;
  for (const auto& [k, v] : tables_) if (!v.empty()) out.push_back(k);
  return out;
}

}  // namespace aster::ingest
