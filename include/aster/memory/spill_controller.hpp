#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"
#include "aster/memory/pool_allocator.hpp"

namespace aster::memory {

enum class SpillKind : uint8_t { HashTable, SortRun, Partition, Other };

struct SpillHandle {
  uint64_t id = 0;
  SpillKind kind = SpillKind::Other;
  size_t bytes = 0;
  uint32_t partition = 0;
  bool on_host = false;
  bool on_nvme = false;
};

// Intermediates spill to pinned host memory first, NVMe second. Partitioned spills let a hash probe
// proceed on the resident partition while the rest load.
class SpillController {
 public:
  SpillController(hal::DevicePtr dev, size_t host_budget, std::string nvme_dir);
  ~SpillController();

  Result<SpillHandle> Spill(const void* device_ptr, size_t bytes, SpillKind kind, uint32_t partition,
                            hal::Stream s = {});
  Result<SpillHandle> SpillHost(const void* host_ptr, size_t bytes, SpillKind kind, uint32_t partition);
  Status Restore(const SpillHandle& h, void* device_dst, hal::Stream s = {});
  Status RestoreToHost(const SpillHandle& h, void* host_dst);
  void Release(const SpillHandle& h);
  void ReleaseAll();
  size_t host_bytes() const;
  size_t nvme_bytes() const;
  uint64_t total_spilled() const { return total_spilled_; }
  std::vector<SpillHandle> handles() const;

 private:
  struct Entry {
    SpillHandle h;
    std::shared_ptr<Buffer> host;
    std::string nvme_path;
  };
  Status DemoteToNvme(Entry& e);
  std::string PathFor(uint64_t id) const;

  hal::DevicePtr dev_;
  size_t host_budget_;
  std::string nvme_dir_;
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, Entry> entries_;
  size_t host_bytes_ = 0;
  size_t nvme_bytes_ = 0;
  uint64_t next_id_ = 1;
  uint64_t total_spilled_ = 0;
};

}  // namespace aster::memory
