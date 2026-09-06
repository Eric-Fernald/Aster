#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/memory/page.hpp"

namespace aster::memory {

// Background worker that pulls pages toward HBM ahead of the tile scheduler.
class Prefetcher {
 public:
  using FetchFn = std::function<Status(PageKey)>;
  Prefetcher(FetchFn fn, uint32_t depth);
  ~Prefetcher();
  void Enqueue(const std::vector<PageKey>& keys);
  void Cancel();
  void Drain();
  size_t pending() const;
  uint32_t depth() const { return depth_; }

 private:
  void Loop();
  FetchFn fetch_;
  uint32_t depth_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<PageKey> queue_;
  size_t in_flight_ = 0;
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace aster::memory
