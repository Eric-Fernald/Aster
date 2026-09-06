#include "aster/memory/residency_map.hpp"

namespace aster::memory {

void ResidencyMap::Register(const PageDesc& d) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  pages_[d.key()] = d;
}

bool ResidencyMap::Has(PageKey k) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  return pages_.count(k) > 0;
}

PageDesc* ResidencyMap::Find(PageKey k) {
  auto it = pages_.find(k);
  return it == pages_.end() ? nullptr : &it->second;
}

const PageDesc* ResidencyMap::Find(PageKey k) const {
  auto it = pages_.find(k);
  return it == pages_.end() ? nullptr : &it->second;
}

Tier ResidencyMap::TierOf(PageKey k) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  const PageDesc* d = Find(k);
  return d ? d->current_tier() : Tier::None;
}

void ResidencyMap::SetTier(PageKey k, Tier t) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (PageDesc* d = Find(k)) d->tier = static_cast<uint8_t>(t);
}

void ResidencyMap::Remove(PageKey k) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  pages_.erase(k);
}

std::vector<PageKey> ResidencyMap::PagesOnTier(Tier t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  std::vector<PageKey> out;
  for (const auto& [k, d] : pages_)
    if (d.current_tier() == t) out.push_back(k);
  return out;
}

std::vector<PageKey> ResidencyMap::Keys() const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  std::vector<PageKey> out;
  out.reserve(pages_.size());
  for (const auto& [k, d] : pages_) out.push_back(k);
  return out;
}

size_t ResidencyMap::BytesOnTier(Tier t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  size_t n = 0;
  for (const auto& [k, d] : pages_)
    if (d.current_tier() == t) n += d.encoded_bytes;
  return n;
}

size_t ResidencyMap::size() const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  return pages_.size();
}

void ResidencyMap::ForEach(const std::function<void(PageDesc&)>& fn) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  for (auto& [k, d] : pages_) fn(d);
}

}  // namespace aster::memory
