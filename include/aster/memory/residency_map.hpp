#pragma once
#include <mutex>
#include <unordered_map>
#include <vector>

#include "aster/memory/page.hpp"

namespace aster::memory {

// Tracks where every registered page currently lives. Shared across GPUs in multi GPU mode.
class ResidencyMap {
 public:
  void Register(const PageDesc& d);
  bool Has(PageKey k) const;
  PageDesc* Find(PageKey k);
  const PageDesc* Find(PageKey k) const;
  Tier TierOf(PageKey k) const;
  void SetTier(PageKey k, Tier t);
  void Remove(PageKey k);
  std::vector<PageKey> PagesOnTier(Tier t) const;
  std::vector<PageKey> Keys() const;
  size_t BytesOnTier(Tier t) const;
  size_t size() const;
  void ForEach(const std::function<void(PageDesc&)>& fn);
  std::recursive_mutex& mutex() const { return mu_; }

 private:
  mutable std::recursive_mutex mu_;
  std::unordered_map<PageKey, PageDesc, PageKeyHash> pages_;
};

}  // namespace aster::memory
