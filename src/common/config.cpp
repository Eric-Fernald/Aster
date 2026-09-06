#include "aster/common/config.hpp"

namespace aster {

const char* HardwareModeName(HardwareMode m) {
  switch (m) {
    case HardwareMode::Discrete: return "discrete";
    case HardwareMode::Coherent: return "coherent";
    default: return "cpu";
  }
}

}  // namespace aster
