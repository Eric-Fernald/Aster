#include "aster/exchange/residency_directory.hpp"

namespace aster::exchange {

void ResidencyDirectory::Attach(int gpu, memory::MemoryManager* mm) {
  std::lock_guard<std::mutex> lk(mu_);
  managers_[gpu] = mm;
}

void ResidencyDirectory::RegisterPartition(const std::string& table, uint32_t partition, int gpu) {
  std::lock_guard<std::mutex> lk(mu_);
  partitions_[{table, partition}] = gpu;
}

int ResidencyDirectory::OwnerOf(const std::string& table, uint32_t partition) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = partitions_.find({table, partition});
  return it == partitions_.end() ? -1 : it->second;
}

std::set<int> ResidencyDirectory::HoldersOf(memory::PageKey key) const {
  std::lock_guard<std::mutex> lk(mu_);
  std::set<int> out;
  for (const auto& [gpu, mm] : managers_)
    if (mm->TierOf(key) == memory::Tier::Hbm) out.insert(gpu);
  return out;
}

int ResidencyDirectory::BestGpuFor(const std::vector<memory::PageKey>& pages) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (managers_.empty()) return -1;
  int best = managers_.begin()->first;
  size_t best_bytes = 0;
  size_t best_free = 0;
  for (const auto& [gpu, mm] : managers_) {
    size_t resident = 0;
    for (const auto& k : pages) {
      std::lock_guard<std::recursive_mutex> rl(mm->residency().mutex());
      const memory::PageDesc* d = mm->residency().Find(k);
      if (d && d->current_tier() == memory::Tier::Hbm) resident += d->encoded_bytes;
    }
    size_t free_b = mm->hbm_free();
    if (resident > best_bytes || (resident == best_bytes && free_b > best_free)) { best = gpu; best_bytes = resident; best_free = free_b; }
  }
  return best;
}

int ResidencyDirectory::num_gpus() const {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<int>(managers_.size());
}

std::vector<int> ResidencyDirectory::gpus() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<int> out;
  for (const auto& [g, m] : managers_) out.push_back(g);
  return out;
}

}  // namespace aster::exchange
