#include "aster/common/log.hpp"

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace aster::log {

static Level g_level = Level::Info;
static std::mutex g_mu;

void SetLevel(Level l) { g_level = l; }
Level GetLevel() { return g_level; }

void Emit(Level l, const std::string& msg) {
  static const char* tag[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::lock_guard<std::mutex> lk(g_mu);
  std::fprintf(stderr, "[aster %s] %s\n", tag[static_cast<int>(l)], msg.c_str());
}

std::string Format(const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

}  // namespace aster::log
