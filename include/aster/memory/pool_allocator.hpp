#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"

namespace aster::memory {

using QueryId = uint64_t;

// Size class pool over a HAL device. Arenas are per query so a finished query returns memory as a unit.
// When RMM is available the pool delegates to rmm::mr::pool_memory_resource.
class PoolAllocator {
 public:
  struct Stats {
    size_t reserved = 0;
    size_t in_use = 0;
    size_t peak = 0;
    uint64_t allocations = 0;
    uint64_t cache_hits = 0;
  };

  PoolAllocator(hal::DevicePtr dev, size_t budget_bytes, MemorySpace space = MemorySpace::Device);
  ~PoolAllocator();

  Result<void*> Allocate(size_t bytes, QueryId q, hal::Stream s = {});
  void Free(void* p, size_t bytes, QueryId q, hal::Stream s = {});
  std::shared_ptr<Buffer> MakeBuffer(size_t bytes, QueryId q, hal::Stream s = {});
  void ReleaseQuery(QueryId q);
  void Trim();
  Stats stats() const;
  size_t budget() const { return budget_; }
  size_t available() const;
  static size_t SizeClass(size_t bytes);

 private:
  struct Block { void* ptr; size_t bytes; };
  Result<void*> RawAlloc(size_t bytes, hal::Stream s);
  void RawFree(void* p, size_t bytes, hal::Stream s);

  hal::DevicePtr dev_;
  size_t budget_;
  MemorySpace space_;
  mutable std::mutex mu_;
  std::map<size_t, std::vector<void*>> free_lists_;
  std::unordered_map<QueryId, std::vector<Block>> arenas_;
  Stats stats_;
};

}  // namespace aster::memory
