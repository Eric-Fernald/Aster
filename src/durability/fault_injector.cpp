#include "aster/durability/fault_injector.hpp"

namespace aster::durability {

FaultInjector& FaultInjector::Global() {
  static FaultInjector f;
  return f;
}

void FaultInjector::Arm(const std::string& point, std::function<void()> action, int fire_after) {
  std::lock_guard<std::mutex> lk(mu_);
  armed_[point] = Entry{std::move(action), fire_after, false};
}

void FaultInjector::Disarm(const std::string& point) {
  std::lock_guard<std::mutex> lk(mu_);
  armed_.erase(point);
}

void FaultInjector::Hit(const char* point) {
  std::function<void()> fire;
  {
    std::lock_guard<std::mutex> lk(mu_);
    ++hits_[point];
    auto it = armed_.find(point);
    if (it == armed_.end() || it->second.fired) return;
    if (it->second.remaining-- > 0) return;
    it->second.fired = true;
    fire = it->second.action;
  }
  if (fire) fire();
}

uint64_t FaultInjector::hits(const std::string& point) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = hits_.find(point);
  return it == hits_.end() ? 0 : it->second;
}

void FaultInjector::Reset() {
  std::lock_guard<std::mutex> lk(mu_);
  armed_.clear();
  hits_.clear();
}

}  // namespace aster::durability
