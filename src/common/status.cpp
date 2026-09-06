#include "aster/common/status.hpp"

namespace aster {

std::string Status::ToString() const {
  static const char* names[] = {"OK", "Invalid", "NotSupported", "OutOfMemory", "IoError", "NotFound", "Internal", "Corrupt"};
  if (ok()) return "OK";
  return std::string(names[static_cast<int>(code_)]) + ": " + msg_;
}

}  // namespace aster
