#pragma once
#include <map>
#include <string>
#include <utility>

#include "aster/common/status.hpp"
#include "aster/hal/device.hpp"
#include "aster/memory/page.hpp"

namespace aster::memory {

// Measured bandwidth per tier pair in bytes per second. The cost model reads this, never a constant.
class BandwidthTable {
 public:
  using Pair = std::pair<Tier, Tier>;
  void Set(Tier from, Tier to, double bytes_per_sec, double latency_us = 0);
  double BytesPerSec(Tier from, Tier to) const;
  double LatencyUs(Tier from, Tier to) const;
  double SecondsFor(Tier from, Tier to, uint64_t bytes) const;
  bool Has(Tier from, Tier to) const;
  Status Save(const std::string& path) const;
  Status Load(const std::string& path);
  std::string ToString() const;
  static BandwidthTable Defaults(bool coherent);

 private:
  struct Entry { double bps; double lat_us; };
  std::map<Pair, Entry> table_;
};

struct ProbeOptions {
  size_t buffer_bytes = size_t(256) << 20;
  int iterations = 5;
  std::string nvme_probe_path;  // empty disables NVMe probe
  bool probe_gds = false;
};

// Benchmarks every reachable tier pair at startup.
Result<BandwidthTable> ProbeBandwidth(hal::Device& dev, const ProbeOptions& opts);
Result<BandwidthTable> LoadOrProbe(hal::Device& dev, const ProbeOptions& opts, const std::string& cache_path);

}  // namespace aster::memory
