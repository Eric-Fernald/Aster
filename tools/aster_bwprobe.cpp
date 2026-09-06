#include <cstdio>
#include <cstring>
#include <string>

#include "aster/hal/device.hpp"
#include "aster/memory/bandwidth_probe.hpp"

// Tier bandwidth microbenchmark. Output feeds the cost model; run once per machine class.
int main(int argc, char** argv) {
  using namespace aster;
  memory::ProbeOptions opts;
  std::string out_path;
  bool cuda = false;
  int device_id = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--cuda") cuda = true;
    else if (a == "--device" && i + 1 < argc) device_id = std::atoi(argv[++i]);
    else if (a == "--bytes" && i + 1 < argc) opts.buffer_bytes = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--iters" && i + 1 < argc) opts.iterations = std::atoi(argv[++i]);
    else if (a == "--nvme" && i + 1 < argc) opts.nvme_probe_path = argv[++i];
    else if (a == "--gds") opts.probe_gds = true;
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else { std::fprintf(stderr, "usage: aster-bwprobe [--cuda] [--device N] [--bytes N] [--iters N] [--nvme PATH] [--gds] [--out FILE]\n"); return 2; }
  }
  auto dev = hal::OpenDevice(cuda ? hal::Backend::Cuda : hal::Backend::Cpu, device_id);
  if (!dev.ok()) { std::fprintf(stderr, "%s\n", dev.status().ToString().c_str()); return 1; }
  auto table = memory::ProbeBandwidth(*dev.value(), opts);
  if (!table.ok()) { std::fprintf(stderr, "%s\n", table.status().ToString().c_str()); return 1; }
  std::printf("%s", table.value().ToString().c_str());
  if (!out_path.empty()) {
    Status s = table.value().Save(out_path);
    if (!s.ok()) { std::fprintf(stderr, "%s\n", s.ToString().c_str()); return 1; }
  }
  return 0;
}
