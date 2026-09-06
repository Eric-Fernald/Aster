#include "aster/exec/fused/kernel_cache.hpp"

#include "aster/memory/gds.hpp"

namespace aster::exec::fused {

KernelCache::KernelCache(std::string dir) : dir_(std::move(dir)) {}

std::string KernelCache::PathFor(const std::string& shape_hash) const { return dir_ + "/" + shape_hash + ".kbin"; }

bool KernelCache::Contains(const std::string& shape_hash) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (mem_.count(shape_hash)) return true;
  return !dir_.empty() && memory::FileExists(PathFor(shape_hash));
}

std::shared_ptr<CompiledKernel> KernelCache::Get(const std::string& shape_hash) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = mem_.find(shape_hash);
  if (it == mem_.end()) { ++misses_; return nullptr; }
  ++hits_;
  return it->second;
}

void KernelCache::Put(std::shared_ptr<CompiledKernel> k) {
  std::lock_guard<std::mutex> lk(mu_);
  mem_[k->shape_hash] = std::move(k);
}

Status KernelCache::Persist(const CompiledKernel& k) const {
  if (dir_.empty()) return Status::OK();
  ASTER_RETURN_NOT_OK(memory::EnsureDir(dir_));
  return memory::WriteFile(PathFor(k.shape_hash), k.image.data(), k.image.size(), false);
}

Result<std::vector<char>> KernelCache::LoadImage(const std::string& shape_hash) const {
  ASTER_ASSIGN_OR_RETURN(auto bytes, memory::ReadWholeFile(PathFor(shape_hash)));
  return std::vector<char>(bytes.begin(), bytes.end());
}

size_t KernelCache::size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return mem_.size();
}

void KernelCache::Clear() {
  std::lock_guard<std::mutex> lk(mu_);
  mem_.clear();
  hits_ = misses_ = 0;
}

}  // namespace aster::exec::fused
