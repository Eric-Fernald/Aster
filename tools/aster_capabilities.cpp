#include <cstdio>
#include <string>

#include "aster/integration/capability_registry.hpp"

// Prints the capability registry (or only its gaps) so coverage is visible and diffable.
int main(int argc, char** argv) {
  using namespace aster::integration;
  CapabilityRegistry reg;
  bool gaps_only = false;
  std::string out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gaps") gaps_only = true;
    else if (a == "--load" && i + 1 < argc) { auto s = reg.LoadTsv(argv[++i]); if (!s.ok()) { std::fprintf(stderr, "%s\n", s.ToString().c_str()); return 1; } }
    else if (a == "--save" && i + 1 < argc) out = argv[++i];
    else { std::fprintf(stderr, "usage: aster-capabilities [--gaps] [--load FILE] [--save FILE]\n"); return 2; }
  }
  if (!out.empty()) { auto s = reg.SaveTsv(out); if (!s.ok()) { std::fprintf(stderr, "%s\n", s.ToString().c_str()); return 1; } return 0; }
  for (const auto& e : gaps_only ? reg.Gaps() : reg.Entries()) std::printf("%-28s %-12s %s\n", e.key.c_str(), SupportName(e.support), e.note.c_str());
  return 0;
}
