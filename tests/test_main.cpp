#include <chrono>
#include <cstring>
#include <filesystem>

#include "aster/common/log.hpp"
#include "test_framework.hpp"

namespace aster_test {
std::string TempDir(const std::string& tag) {
  auto p = std::filesystem::temp_directory_path() / ("aster_test_" + tag + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(p);
  return p.string();
}
}  // namespace aster_test

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  aster::log::SetLevel(argc > 2 && std::strcmp(argv[2], "-v") == 0 ? aster::log::Level::Debug : aster::log::Level::Warn);
  int run = 0, failed = 0;
  for (const auto& c : aster_test::Registry::cases()) {
    if (filter && std::strstr(c.name, filter) == nullptr) continue;
    ++run;
    auto t0 = std::chrono::steady_clock::now();
    try {
      c.fn();
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("[ OK ] %s (%.1f ms)\n", c.name, ms);
    } catch (const aster_test::Failure& f) {
      ++failed;
      std::printf("[FAIL] %s\n       %s\n", c.name, f.msg.c_str());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[FAIL] %s\n       exception: %s\n", c.name, e.what());
    }
  }
  std::printf("%d tests, %d failed\n", run, failed);
  return failed ? 1 : 0;
}
