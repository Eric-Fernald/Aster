#pragma once
#include <string>

namespace aster::log {

enum class Level { Debug, Info, Warn, Error };
void SetLevel(Level l);
Level GetLevel();
void Emit(Level l, const std::string& msg);
std::string Format(const char* fmt, ...);

}  // namespace aster::log

#define ASTER_LOG(level, ...)                                                                    \
  do {                                                                                           \
    if (::aster::log::Level::level >= ::aster::log::GetLevel())                                  \
      ::aster::log::Emit(::aster::log::Level::level, ::aster::log::Format(__VA_ARGS__));         \
  } while (0)
