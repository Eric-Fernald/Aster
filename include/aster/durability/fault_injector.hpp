#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aster::durability {

// Named kill points for chaos tests: mid WAL write, mid compaction, mid query. A registered action
// fires once when the point is reached (throw, abort, or flip a flag) and the test verifies recovery.
class FaultInjector {
 public:
  static FaultInjector& Global();
  void Arm(const std::string& point, std::function<void()> action, int fire_after = 0);
  void Disarm(const std::string& point);
  void Hit(const char* point);
  uint64_t hits(const std::string& point) const;
  void Reset();

 private:
  struct Entry { std::function<void()> action; int remaining; bool fired; };
  mutable std::mutex mu_;
  std::unordered_map<std::string, Entry> armed_;
  std::unordered_map<std::string, uint64_t> hits_;
};

#define ASTER_FAULT_POINT(name) ::aster::durability::FaultInjector::Global().Hit(name)

}  // namespace aster::durability
