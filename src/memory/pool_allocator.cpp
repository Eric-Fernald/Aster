#include "aster/memory/pool_allocator.hpp"

#include <algorithm>

namespace aster::memory {

namespace {
struct PoolCtx {
  PoolAllocator* pool;
  QueryId q;
};
void PoolDeleter(void* p, size_t bytes, void* ctx) {
  auto* c = static_cast<PoolCtx*>(ctx);
  c->pool->Free(p, bytes, c->q);
  delete c;
}
}  // namespace

PoolAllocator::PoolAllocator(hal::DevicePtr dev, size_t budget_bytes, MemorySpace space)
    : dev_(std::move(dev)), budget_(budget_bytes), space_(space) {}

PoolAllocator::~PoolAllocator() {
  for (auto& [q, blocks] : arenas_)
    for (auto& b : blocks) RawFree(b.ptr, b.bytes, {});
  Trim();
}

size_t PoolAllocator::SizeClass(size_t bytes) {
  if (bytes <= 256) return 256;
  size_t c = 256;
  while (c < bytes) c <<= 1;
  // Above 64 MB use 16 MB granularity to limit waste on large chunks.
  if (c > (size_t(64) << 20)) c = (bytes + (size_t(16) << 20) - 1) & ~((size_t(16) << 20) - 1);
  return c;
}

Result<void*> PoolAllocator::RawAlloc(size_t bytes, hal::Stream s) {
  switch (space_) {
    case MemorySpace::Device: return dev_->AllocateDevice(bytes, s);
    case MemorySpace::HostPinned: return dev_->AllocatePinned(bytes);
    case MemorySpace::Managed: return dev_->AllocateManaged(bytes);
    default: return dev_->AllocatePinned(bytes);
  }
}

void PoolAllocator::RawFree(void* p, size_t bytes, hal::Stream s) {
  switch (space_) {
    case MemorySpace::Device: dev_->FreeDevice(p, bytes, s); break;
    case MemorySpace::HostPinned: dev_->FreePinned(p, bytes); break;
    case MemorySpace::Managed: dev_->FreeManaged(p, bytes); break;
    default: dev_->FreePinned(p, bytes); break;
  }
}

Result<void*> PoolAllocator::Allocate(size_t bytes, QueryId q, hal::Stream s) {
  size_t cls = SizeClass(bytes);
  std::lock_guard<std::mutex> lk(mu_);
  ++stats_.allocations;
  void* p = nullptr;
  auto& fl = free_lists_[cls];
  if (!fl.empty()) {
    p = fl.back();
    fl.pop_back();
    ++stats_.cache_hits;
  } else {
    if (budget_ && stats_.reserved + cls > budget_) {
      // Trim idle blocks before giving up.
      for (auto& [c, v] : free_lists_) {
        for (void* fp : v) { RawFree(fp, c, s); stats_.reserved -= c; }
        v.clear();
      }
      if (stats_.reserved + cls > budget_) return Status::OutOfMemory("pool budget exceeded");
    }
    ASTER_ASSIGN_OR_RETURN(p, RawAlloc(cls, s));
    stats_.reserved += cls;
  }
  stats_.in_use += cls;
  stats_.peak = std::max(stats_.peak, stats_.in_use);
  arenas_[q].push_back({p, cls});
  return p;
}

void PoolAllocator::Free(void* p, size_t bytes, QueryId q, hal::Stream) {
  size_t cls = SizeClass(bytes);
  std::lock_guard<std::mutex> lk(mu_);
  auto it = arenas_.find(q);
  if (it != arenas_.end()) {
    auto& v = it->second;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const Block& b) { return b.ptr == p; }), v.end());
  }
  stats_.in_use -= std::min(stats_.in_use, cls);
  free_lists_[cls].push_back(p);
}

std::shared_ptr<Buffer> PoolAllocator::MakeBuffer(size_t bytes, QueryId q, hal::Stream s) {
  auto r = Allocate(bytes, q, s);
  if (!r.ok()) return nullptr;
  return std::make_shared<Buffer>(r.value(), bytes, space_, PoolDeleter, new PoolCtx{this, q});
}

void PoolAllocator::ReleaseQuery(QueryId q) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = arenas_.find(q);
  if (it == arenas_.end()) return;
  for (auto& b : it->second) {
    stats_.in_use -= std::min(stats_.in_use, b.bytes);
    free_lists_[b.bytes].push_back(b.ptr);
  }
  arenas_.erase(it);
}

void PoolAllocator::Trim() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [c, v] : free_lists_) {
    for (void* p : v) { RawFree(p, c, {}); stats_.reserved -= std::min(stats_.reserved, c); }
    v.clear();
  }
}

PoolAllocator::Stats PoolAllocator::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

size_t PoolAllocator::available() const {
  std::lock_guard<std::mutex> lk(mu_);
  return budget_ > stats_.in_use ? budget_ - stats_.in_use : 0;
}

}  // namespace aster::memory
