#include "aster/memory/page.hpp"

namespace aster::memory {

const char* TierName(Tier t) {
  switch (t) {
    case Tier::Hbm: return "hbm";
    case Tier::Host: return "host";
    case Tier::Nvme: return "nvme";
    case Tier::Object: return "object";
    default: return "none";
  }
}

}  // namespace aster::memory
