#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aster/common/status.hpp"

namespace aster::exec::fused {

struct CompiledKernel {
  std::string shape_hash;
  std::string name;
  std::vector<char> image;   // cubin or ptx
  void* module = nullptr;    // CUmodule
  void* function = nullptr;  // CUfunction
  double compile_ms = 0;
  bool from_disk = false;
};

// Keyed by pipeline shape, not query text, so structurally similar queries reuse compiled code.
// Memory tier first, then an on disk directory that survives restarts.
class KernelCache {
 public:
  explicit KernelCache(std::string dir);
  bool Contains(const std::string& shape_hash) const;
  std::shared_ptr<CompiledKernel> Get(const std::string& shape_hash) const;
  void Put(std::shared_ptr<CompiledKernel> k);
  Status Persist(const CompiledKernel& k) const;
  Result<std::vector<char>> LoadImage(const std::string& shape_hash) const;
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  size_t size() const;
  void Clear();

 private:
  std::string PathFor(const std::string& shape_hash) const;
  std::string dir_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<CompiledKernel>> mem_;
  mutable uint64_t hits_ = 0, misses_ = 0;
};

}  // namespace aster::exec::fused
