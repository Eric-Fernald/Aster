#include "aster/memory/prefetcher.hpp"

#include "aster/common/log.hpp"

namespace aster::memory {

Prefetcher::Prefetcher(FetchFn fn, uint32_t depth) : fetch_(std::move(fn)), depth_(depth ? depth : 1) {
  worker_ = std::thread([this] { Loop(); });
}

Prefetcher::~Prefetcher() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    queue_.clear();
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void Prefetcher::Enqueue(const std::vector<PageKey>& keys) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& k : keys) queue_.push_back(k);
  }
  cv_.notify_all();
}

void Prefetcher::Cancel() {
  std::lock_guard<std::mutex> lk(mu_);
  queue_.clear();
}

void Prefetcher::Drain() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [this] { return queue_.empty() && in_flight_ == 0; });
}

size_t Prefetcher::pending() const {
  std::lock_guard<std::mutex> lk(mu_);
  return queue_.size() + in_flight_;
}

void Prefetcher::Loop() {
  for (;;) {
    PageKey key;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      key = queue_.front();
      queue_.pop_front();
      ++in_flight_;
    }
    Status s = fetch_(key);
    if (!s.ok()) ASTER_LOG(Debug, "prefetch %s failed: %s", key.ToString().c_str(), s.ToString().c_str());
    {
      std::lock_guard<std::mutex> lk(mu_);
      --in_flight_;
    }
    cv_.notify_all();
  }
}

}  // namespace aster::memory
