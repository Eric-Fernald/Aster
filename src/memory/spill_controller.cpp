#include "aster/memory/spill_controller.hpp"

#include <cstdio>
#include <cstring>

#include "aster/common/log.hpp"
#include "aster/memory/gds.hpp"

namespace aster::memory {

SpillController::SpillController(hal::DevicePtr dev, size_t host_budget, std::string nvme_dir)
    : dev_(std::move(dev)), host_budget_(host_budget), nvme_dir_(std::move(nvme_dir)) {}

SpillController::~SpillController() { ReleaseAll(); }

std::string SpillController::PathFor(uint64_t id) const { return nvme_dir_ + "/spill_" + std::to_string(id) + ".bin"; }

Result<SpillHandle> SpillController::Spill(const void* device_ptr, size_t bytes, SpillKind kind, uint32_t partition,
                                           hal::Stream s) {
  auto host = dev_->MakePinnedBuffer(bytes);
  if (!host) host = Buffer::AllocateHost(bytes);
  ASTER_RETURN_NOT_OK(dev_->CopyDeviceToHost(host->data(), device_ptr, bytes, s));
  ASTER_RETURN_NOT_OK(dev_->Synchronize(s));
  std::lock_guard<std::mutex> lk(mu_);
  Entry e;
  e.h = {next_id_++, kind, bytes, partition, true, false};
  e.host = host;
  host_bytes_ += bytes;
  total_spilled_ += bytes;
  while (host_bytes_ > host_budget_) {
    // Demote the oldest host resident entry that is not this one.
    Entry* victim = nullptr;
    for (auto& [id, en] : entries_)
      if (en.h.on_host && !en.h.on_nvme && (!victim || id < victim->h.id)) victim = &en;
    if (!victim) break;
    ASTER_RETURN_NOT_OK(DemoteToNvme(*victim));
  }
  SpillHandle h = e.h;
  entries_[h.id] = std::move(e);
  return h;
}

Result<SpillHandle> SpillController::SpillHost(const void* host_ptr, size_t bytes, SpillKind kind, uint32_t partition) {
  auto host = Buffer::CopyOf(host_ptr, bytes);
  std::lock_guard<std::mutex> lk(mu_);
  Entry e;
  e.h = {next_id_++, kind, bytes, partition, true, false};
  e.host = host;
  host_bytes_ += bytes;
  total_spilled_ += bytes;
  if (host_bytes_ > host_budget_) ASTER_RETURN_NOT_OK(DemoteToNvme(e));
  SpillHandle h = e.h;
  entries_[h.id] = std::move(e);
  return h;
}

Status SpillController::DemoteToNvme(Entry& e) {
  if (!e.h.on_host) return Status::OK();
  ASTER_RETURN_NOT_OK(EnsureDir(nvme_dir_));
  e.nvme_path = PathFor(e.h.id);
  ASTER_RETURN_NOT_OK(WriteFile(e.nvme_path, e.host->data(), e.h.bytes, false));
  host_bytes_ -= e.h.bytes;
  nvme_bytes_ += e.h.bytes;
  e.host.reset();
  e.h.on_host = false;
  e.h.on_nvme = true;
  ASTER_LOG(Debug, "spill %llu demoted to nvme (%zu bytes)", (unsigned long long)e.h.id, e.h.bytes);
  return Status::OK();
}

Status SpillController::RestoreToHost(const SpillHandle& h, void* host_dst) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(h.id);
  if (it == entries_.end()) return Status::NotFound("spill " + std::to_string(h.id));
  Entry& e = it->second;
  if (e.h.on_host) {
    std::memcpy(host_dst, e.host->data(), e.h.bytes);
    return Status::OK();
  }
  PosixFileReader r(dev_);
  return r.ReadToHost(e.nvme_path, 0, static_cast<uint32_t>(e.h.bytes), host_dst);
}

Status SpillController::Restore(const SpillHandle& h, void* device_dst, hal::Stream s) {
  std::unique_lock<std::mutex> lk(mu_);
  auto it = entries_.find(h.id);
  if (it == entries_.end()) return Status::NotFound("spill " + std::to_string(h.id));
  Entry& e = it->second;
  if (e.h.on_host) {
    ASTER_RETURN_NOT_OK(dev_->CopyHostToDevice(device_dst, e.host->data(), e.h.bytes, s));
    return dev_->Synchronize(s);
  }
  std::string path = e.nvme_path;
  size_t bytes = e.h.bytes;
  lk.unlock();
  auto reader = MakeFileReader(dev_, true);
  return reader->ReadToDevice(path, 0, static_cast<uint32_t>(bytes), device_dst, s);
}

void SpillController::Release(const SpillHandle& h) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = entries_.find(h.id);
  if (it == entries_.end()) return;
  Entry& e = it->second;
  if (e.h.on_host) host_bytes_ -= e.h.bytes;
  if (e.h.on_nvme) { nvme_bytes_ -= e.h.bytes; std::remove(e.nvme_path.c_str()); }
  entries_.erase(it);
}

void SpillController::ReleaseAll() {
  std::vector<SpillHandle> hs = handles();
  for (const auto& h : hs) Release(h);
}

size_t SpillController::host_bytes() const { std::lock_guard<std::mutex> lk(mu_); return host_bytes_; }
size_t SpillController::nvme_bytes() const { std::lock_guard<std::mutex> lk(mu_); return nvme_bytes_; }

std::vector<SpillHandle> SpillController::handles() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<SpillHandle> out;
  for (const auto& [id, e] : entries_) out.push_back(e.h);
  return out;
}

}  // namespace aster::memory
