#include "aster/memory/memory_manager.hpp"

#include <algorithm>
#include <cstring>

#include "aster/common/hash.hpp"
#include "aster/common/log.hpp"
#include "aster/metrics/metrics.hpp"

namespace aster::memory {

namespace {
// Budgets are tracked in pool size classes so accounting matches what the allocator reserves.
inline size_t Cls(size_t bytes) { return PoolAllocator::SizeClass(bytes); }
}  // namespace

PageHandle::~PageHandle() {
  if (mm_ && ptr_) mm_->Release(key_);
}

PageHandle& PageHandle::operator=(PageHandle&& o) noexcept {
  if (this != &o) {
    if (mm_ && ptr_) mm_->Release(key_);
    mm_ = o.mm_; key_ = o.key_; ptr_ = o.ptr_; bytes_ = o.bytes_; space_ = o.space_;
    o.mm_ = nullptr; o.ptr_ = nullptr;
  }
  return *this;
}

MemoryManager::MemoryManager(const EngineConfig& cfg, hal::DevicePtr dev, BandwidthTable bw)
    : cfg_(cfg), dev_(std::move(dev)), bw_(std::move(bw)) {
  coherent_ = cfg.mode == HardwareMode::Coherent && dev_->info().coherent_host_access;
  if (cfg.mode == HardwareMode::Coherent && !coherent_)
    ASTER_LOG(Warn, "coherent mode requested but device lacks pageable memory access; using discrete mode");
  hbm_budget_ = cfg.vram_budget_bytes ? cfg.vram_budget_bytes : static_cast<size_t>(dev_->info().free_memory * 0.9);
  device_pool_ = std::make_unique<PoolAllocator>(dev_, hbm_budget_, MemorySpace::Device);
  host_pool_ = std::make_unique<PoolAllocator>(dev_, cfg.host_pinned_budget_bytes,
                                               coherent_ ? MemorySpace::Managed : MemorySpace::HostPinned);
  reader_ = MakeFileReader(dev_, cfg.enable_gds);
  spill_ = std::make_unique<SpillController>(dev_, cfg.host_pinned_budget_bytes / 4, cfg.nvme_spill_dir);
  prefetcher_ = std::make_unique<Prefetcher>([this](PageKey k) { return Fetch(k); }, cfg.prefetch_depth);
  ASTER_LOG(Info, "memory manager: mode=%s hbm_budget=%zu MB reader=%s", coherent_ ? "coherent" : "discrete",
            hbm_budget_ >> 20, reader_->name());
}

MemoryManager::~MemoryManager() {
  prefetcher_.reset();
  residency_.ForEach([&](PageDesc& d) { FreeDevicePage(d); FreeHostPage(d); });
}

void MemoryManager::RegisterPage(const PageDesc& d) { residency_.Register(d); }

void MemoryManager::UnregisterSegment(SegmentId seg) {
  std::scoped_lock lk(mu_, residency_.mutex());
  std::vector<PageKey> keys;
  residency_.ForEach([&](PageDesc& d) {
    if (d.segment_id == seg) { FreeDevicePage(d); FreeHostPage(d); keys.push_back(d.key()); }
  });
  for (const auto& k : keys) residency_.Remove(k);
}

Tier MemoryManager::TierOf(PageKey key) const { return residency_.TierOf(key); }

size_t MemoryManager::hbm_free() const {
  std::scoped_lock lk(mu_, residency_.mutex());
  return hbm_budget_ > hbm_used_ ? hbm_budget_ - hbm_used_ : 0;
}

MemoryStats MemoryManager::stats() const {
  std::scoped_lock lk(mu_, residency_.mutex());
  MemoryStats s = stats_;
  s.hbm_in_use = hbm_used_;
  s.host_in_use = host_used_;
  return s;
}

Status MemoryManager::VerifyChecksum(const PageDesc& d, const void* host_data) {
  if (d.locator.checksum == 0) return Status::OK();
  uint32_t crc = Crc32c(host_data, d.encoded_bytes);
  if (crc != d.locator.checksum)
    return Status::Corrupt("checksum mismatch on page " + d.key().ToString());
  return Status::OK();
}

void MemoryManager::FreeDevicePage(PageDesc& d) {
  if (!d.device_ptr) return;
  if (!coherent_) {
    device_pool_->Free(d.device_ptr, d.encoded_bytes, 0);
    hbm_used_ -= std::min(hbm_used_, Cls(d.encoded_bytes));
  }
  d.device_ptr = nullptr;
}

void MemoryManager::FreeHostPage(PageDesc& d) {
  if (!d.host_ptr) return;
  host_pool_->Free(d.host_ptr, d.encoded_bytes, 0);
  host_used_ -= std::min(host_used_, Cls(d.encoded_bytes));
  d.host_ptr = nullptr;
}

Status MemoryManager::LoadToHost(PageDesc& d) {
  if (d.host_ptr) return Status::OK();
  if (host_used_ + Cls(d.encoded_bytes) > cfg_.host_pinned_budget_bytes) {
    auto victims = eviction_.SelectVictims(residency_, Tier::Host, d.encoded_bytes, epoch_);
    for (const auto& k : victims) {
      if (PageDesc* v = residency_.Find(k)) { FreeHostPage(*v); v->tier = static_cast<uint8_t>(Tier::Nvme); ++stats_.evictions; }
    }
  }
  ASTER_ASSIGN_OR_RETURN(d.host_ptr, host_pool_->Allocate(d.encoded_bytes, 0));
  Status s = reader_->ReadToHost(d.locator.path, d.locator.offset, d.encoded_bytes, d.host_ptr);
  if (s.ok()) s = VerifyChecksum(d, d.host_ptr);
  if (!s.ok()) { host_pool_->Free(d.host_ptr, d.encoded_bytes, 0); d.host_ptr = nullptr; return s; }
  host_used_ += Cls(d.encoded_bytes);
  stats_.bytes_nvme_to_host += d.encoded_bytes;
  metrics::Registry::Global().counter("bytes_nvme_to_host").Add(d.encoded_bytes);
  if (d.current_tier() > Tier::Host) d.tier = static_cast<uint8_t>(Tier::Host);
  return Status::OK();
}

Status MemoryManager::EnsureFree(size_t bytes) {
  bytes = Cls(bytes);
  if (hbm_used_ + bytes <= hbm_budget_) return Status::OK();
  size_t need = hbm_used_ + bytes - hbm_budget_;
  auto victims = eviction_.SelectVictims(residency_, Tier::Hbm, need, epoch_);
  if (victims.empty()) return Status::OutOfMemory("no evictable HBM pages");
  size_t freed = 0;
  for (const auto& k : victims) {
    PageDesc* v = residency_.Find(k);
    if (!v) continue;
    freed += v->encoded_bytes;
    ASTER_RETURN_NOT_OK(Evict(k));
  }
  if (freed < need) return Status::OutOfMemory("evicted " + std::to_string(freed) + " of " + std::to_string(need));
  return Status::OK();
}

Status MemoryManager::PromoteHostToDevice(PageDesc& d, hal::Stream s) {
  if (coherent_) {
    // Host memory is already device accessible; residency in HBM is a placement hint.
    dev_->AdviseAccess(d.host_ptr, d.encoded_bytes, true);
    dev_->PrefetchManaged(d.host_ptr, d.encoded_bytes, s);
    d.device_ptr = d.host_ptr;
    hbm_used_ += Cls(d.encoded_bytes);
    stats_.bytes_host_to_hbm += d.encoded_bytes;
    d.tier = static_cast<uint8_t>(Tier::Hbm);
    return Status::OK();
  }
  ASTER_RETURN_NOT_OK(EnsureFree(d.encoded_bytes));
  ASTER_ASSIGN_OR_RETURN(d.device_ptr, device_pool_->Allocate(d.encoded_bytes, 0, s));
  Status st = dev_->CopyHostToDevice(d.device_ptr, d.host_ptr, d.encoded_bytes, s);
  if (st.ok()) st = dev_->Synchronize(s);
  if (!st.ok()) { device_pool_->Free(d.device_ptr, d.encoded_bytes, 0, s); d.device_ptr = nullptr; return st; }
  hbm_used_ += Cls(d.encoded_bytes);
  stats_.bytes_host_to_hbm += d.encoded_bytes;
  metrics::Registry::Global().counter("bytes_host_to_hbm").Add(d.encoded_bytes);
  d.tier = static_cast<uint8_t>(Tier::Hbm);
  return Status::OK();
}

Status MemoryManager::LoadToDevice(PageDesc& d, hal::Stream s) {
  if (d.device_ptr) return Status::OK();
  if (d.host_ptr || coherent_ || !reader_->direct_to_device()) {
    ASTER_RETURN_NOT_OK(LoadToHost(d));
    return PromoteHostToDevice(d, s);
  }
  // GDS path: NVMe straight into VRAM. Checksum is verified by the decode kernel.
  ASTER_RETURN_NOT_OK(EnsureFree(d.encoded_bytes));
  ASTER_ASSIGN_OR_RETURN(d.device_ptr, device_pool_->Allocate(d.encoded_bytes, 0, s));
  Status st = reader_->ReadToDevice(d.locator.path, d.locator.offset, d.encoded_bytes, d.device_ptr, s);
  if (!st.ok()) { device_pool_->Free(d.device_ptr, d.encoded_bytes, 0, s); d.device_ptr = nullptr; return st; }
  hbm_used_ += Cls(d.encoded_bytes);
  stats_.bytes_nvme_to_hbm += d.encoded_bytes;
  metrics::Registry::Global().counter("bytes_nvme_to_hbm").Add(d.encoded_bytes);
  d.tier = static_cast<uint8_t>(Tier::Hbm);
  return Status::OK();
}

Result<PageHandle> MemoryManager::Acquire(PageKey key, hal::Stream s) {
  std::scoped_lock lk(mu_, residency_.mutex());
  PageDesc* d = residency_.Find(key);
  if (!d) return Status::NotFound("page " + key.ToString());
  ++epoch_;
  bool was_resident = d->device_ptr != nullptr;
  if (!was_resident) {
    Status st = LoadToDevice(*d, s);
    if (!st.ok()) return st;
    if (coherent_) ++stats_.page_faults;
    ++stats_.prefetch_misses;
  } else {
    ++stats_.prefetch_hits;
  }
  d->pin_count++;
  d->access_count++;
  d->last_access_epoch = epoch_;
  MemorySpace space = coherent_ ? MemorySpace::Managed : MemorySpace::Device;
  if (dev_->backend() == hal::Backend::Cpu) space = MemorySpace::Host;
  return PageHandle(this, key, d->device_ptr, d->encoded_bytes, space);
}

void MemoryManager::Release(PageKey key) {
  std::scoped_lock lk(mu_, residency_.mutex());
  if (PageDesc* d = residency_.Find(key))
    if (d->pin_count > 0) d->pin_count--;
}

Status MemoryManager::Fetch(PageKey key, hal::Stream s) {
  std::scoped_lock lk(mu_, residency_.mutex());
  PageDesc* d = residency_.Find(key);
  if (!d) return Status::NotFound("page " + key.ToString());
  if (d->device_ptr) return Status::OK();
  ++epoch_;
  d->last_access_epoch = epoch_;
  return LoadToDevice(*d, s);
}

void MemoryManager::Prefetch(const std::vector<PageKey>& keys) { prefetcher_->Enqueue(keys); }

void MemoryManager::MarkScheduled(const std::vector<PageKey>& keys, bool scheduled) {
  std::scoped_lock lk(mu_, residency_.mutex());
  for (const auto& k : keys)
    if (PageDesc* d = residency_.Find(k)) d->scheduled = scheduled;
}

Status MemoryManager::Evict(PageKey key) {
  PageDesc* d = residency_.Find(key);
  if (!d) return Status::NotFound("page " + key.ToString());
  if (d->pin_count > 0) return Status::Invalid("page pinned: " + key.ToString());
  if (!d->device_ptr) return Status::OK();
  if (coherent_) {
    dev_->AdviseAccess(d->host_ptr, d->encoded_bytes, false);
    d->device_ptr = nullptr;
    hbm_used_ -= std::min(hbm_used_, Cls(d->encoded_bytes));
  } else {
    FreeDevicePage(*d);
    stats_.bytes_hbm_to_host += 0;  // pages are immutable; dropping needs no writeback
  }
  d->tier = static_cast<uint8_t>(d->host_ptr ? Tier::Host : Tier::Nvme);
  ++stats_.evictions;
  metrics::Registry::Global().counter("evictions").Add(1);
  return Status::OK();
}

}  // namespace aster::memory
