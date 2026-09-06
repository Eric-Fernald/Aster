#pragma once
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "aster/memory/memory_manager.hpp"
#include "aster/memory/page.hpp"

namespace aster::exchange {

// Shared residency map across GPUs. The planner places a pipeline on the GPU that already holds
// most of its input pages; partitioned loads register which GPU owns which partition.
class ResidencyDirectory {
 public:
  void Attach(int gpu, memory::MemoryManager* mm);
  void RegisterPartition(const std::string& table, uint32_t partition, int gpu);
  int OwnerOf(const std::string& table, uint32_t partition) const;
  std::set<int> HoldersOf(memory::PageKey key) const;
  // GPU with the most bytes of `pages` resident in HBM; ties go to the least loaded GPU.
  int BestGpuFor(const std::vector<memory::PageKey>& pages) const;
  int num_gpus() const;
  std::vector<int> gpus() const;

 private:
  mutable std::mutex mu_;
  std::map<int, memory::MemoryManager*> managers_;
  std::map<std::pair<std::string, uint32_t>, int> partitions_;
};

}  // namespace aster::exchange
