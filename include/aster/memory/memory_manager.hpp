#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/config.hpp"
#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"
#include "aster/memory/bandwidth_probe.hpp"
#include "aster/memory/eviction.hpp"
#include "aster/memory/gds.hpp"
#include "aster/memory/page.hpp"
#include "aster/memory/pool_allocator.hpp"
#include "aster/memory/prefetcher.hpp"
#include "aster/memory/residency_map.hpp"
#include "aster/memory/spill_controller.hpp"

namespace aster::memory {

// A pinned, device accessible view of a page. Released on destruction.
class PageHandle {
 public:
  PageHandle() = default;
  PageHandle(class MemoryManager* mm, PageKey key, const void* ptr, uint32_t bytes, MemorySpace space)
      : mm_(mm), key_(key), ptr_(ptr), bytes_(bytes), space_(space) {}
  ~PageHandle();
  PageHandle(PageHandle&& o) noexcept { *this = std::move(o); }
  PageHandle& operator=(PageHandle&& o) noexcept;
  PageHandle(const PageHandle&) = delete;
  PageHandle& operator=(const PageHandle&) = delete;

  const void* data() const { return ptr_; }
  uint32_t bytes() const { return bytes_; }
  MemorySpace space() const { return space_; }
  PageKey key() const { return key_; }
  bool valid() const { return ptr_ != nullptr; }

 private:
  MemoryManager* mm_ = nullptr;
  PageKey key_;
  const void* ptr_ = nullptr;
  uint32_t bytes_ = 0;
  MemorySpace space_ = MemorySpace::Host;
};

struct MemoryStats {
  uint64_t bytes_nvme_to_host = 0;
  uint64_t bytes_nvme_to_hbm = 0;
  uint64_t bytes_host_to_hbm = 0;
  uint64_t bytes_hbm_to_host = 0;
  uint64_t evictions = 0;
  uint64_t page_faults = 0;   // coherent mode on demand pulls
  uint64_t prefetch_hits = 0;
  uint64_t prefetch_misses = 0;
  size_t hbm_in_use = 0;
  size_t host_in_use = 0;
};

// The core bet: HBM is the top tier of a coherent hierarchy, not a copy destination.
// Discrete mode moves pages with explicit copies. Coherent mode maps host memory into the GPU
// address space and treats HBM residency as a placement hint.
class MemoryManager {
 public:
  MemoryManager(const EngineConfig& cfg, hal::DevicePtr dev, BandwidthTable bw);
  ~MemoryManager();

  void RegisterPage(const PageDesc& d);
  void UnregisterSegment(SegmentId seg);
  bool coherent() const { return coherent_; }
  hal::Device& device() { return *dev_; }
  const BandwidthTable& bandwidth() const { return bw_; }
  ResidencyMap& residency() { return residency_; }
  PoolAllocator& device_pool() { return *device_pool_; }
  PoolAllocator& host_pool() { return *host_pool_; }
  SpillController& spill() { return *spill_; }

  // Ensures the page is device accessible and pinned until the handle drops.
  Result<PageHandle> Acquire(PageKey key, hal::Stream s = {});
  // Moves a page toward HBM without pinning it.
  Status Fetch(PageKey key, hal::Stream s = {});
  void Prefetch(const std::vector<PageKey>& keys);
  void MarkScheduled(const std::vector<PageKey>& keys, bool scheduled);
  Status Evict(PageKey key);
  Status EnsureFree(size_t bytes);
  Tier TierOf(PageKey key) const;
  size_t hbm_budget() const { return hbm_budget_; }
  size_t hbm_free() const;
  MemoryStats stats() const;
  uint64_t epoch() const { return epoch_; }

 private:
  friend class PageHandle;
  void Release(PageKey key);
  Status LoadToHost(PageDesc& d);
  Status LoadToDevice(PageDesc& d, hal::Stream s);
  Status PromoteHostToDevice(PageDesc& d, hal::Stream s);
  Status VerifyChecksum(const PageDesc& d, const void* host_data);
  void FreeDevicePage(PageDesc& d);
  void FreeHostPage(PageDesc& d);

  EngineConfig cfg_;
  hal::DevicePtr dev_;
  BandwidthTable bw_;
  bool coherent_;
  size_t hbm_budget_;
  ResidencyMap residency_;
  EvictionPolicy eviction_;
  std::unique_ptr<PoolAllocator> device_pool_;
  std::unique_ptr<PoolAllocator> host_pool_;
  std::unique_ptr<FileReader> reader_;
  std::unique_ptr<SpillController> spill_;
  std::unique_ptr<Prefetcher> prefetcher_;
  mutable std::mutex mu_;
  MemoryStats stats_;
  uint64_t epoch_ = 1;
  size_t hbm_used_ = 0;
  size_t host_used_ = 0;
};

}  // namespace aster::memory
