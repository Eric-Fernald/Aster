#include "aster/memory/bandwidth_probe.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#include "aster/common/log.hpp"
#include "aster/memory/gds.hpp"

namespace aster::memory {

void BandwidthTable::Set(Tier from, Tier to, double bps, double lat_us) { table_[{from, to}] = {bps, lat_us}; }

bool BandwidthTable::Has(Tier from, Tier to) const { return table_.count({from, to}) > 0; }

double BandwidthTable::BytesPerSec(Tier from, Tier to) const {
  if (from == to) return 1e15;
  auto it = table_.find({from, to});
  if (it != table_.end()) return it->second.bps;
  // Multi hop: chain through intermediate tiers, bottleneck wins.
  double best = 0;
  for (int mid = 0; mid < 4; ++mid) {
    Tier m = static_cast<Tier>(mid);
    if (m == from || m == to) continue;
    auto a = table_.find({from, m});
    auto b = table_.find({m, to});
    if (a != table_.end() && b != table_.end()) best = std::max(best, std::min(a->second.bps, b->second.bps));
  }
  return best > 0 ? best : 1e9;
}

double BandwidthTable::LatencyUs(Tier from, Tier to) const {
  auto it = table_.find({from, to});
  return it == table_.end() ? 0.0 : it->second.lat_us;
}

double BandwidthTable::SecondsFor(Tier from, Tier to, uint64_t bytes) const {
  if (from == to || bytes == 0) return 0.0;
  return double(bytes) / BytesPerSec(from, to) + LatencyUs(from, to) * 1e-6;
}

Status BandwidthTable::Save(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return Status::IoError("cannot write " + path);
  for (const auto& [k, v] : table_)
    f << int(k.first) << "\t" << int(k.second) << "\t" << v.bps << "\t" << v.lat_us << "\n";
  return Status::OK();
}

Status BandwidthTable::Load(const std::string& path) {
  std::ifstream f(path);
  if (!f) return Status::NotFound(path);
  int a, b;
  double bps, lat;
  while (f >> a >> b >> bps >> lat) Set(static_cast<Tier>(a), static_cast<Tier>(b), bps, lat);
  return table_.empty() ? Status::Corrupt("empty bandwidth table") : Status::OK();
}

std::string BandwidthTable::ToString() const {
  std::ostringstream os;
  for (const auto& [k, v] : table_)
    os << TierName(k.first) << "->" << TierName(k.second) << ": " << v.bps / 1e9 << " GB/s (" << v.lat_us << " us)\n";
  return os.str();
}

BandwidthTable BandwidthTable::Defaults(bool coherent) {
  BandwidthTable t;
  double h2d = coherent ? 900e9 : 25e9;
  t.Set(Tier::Host, Tier::Hbm, h2d, coherent ? 2 : 10);
  t.Set(Tier::Hbm, Tier::Host, h2d, coherent ? 2 : 10);
  t.Set(Tier::Hbm, Tier::Hbm, 3000e9, 0);
  t.Set(Tier::Nvme, Tier::Host, 6e9, 80);
  t.Set(Tier::Nvme, Tier::Hbm, 6e9, 100);
  t.Set(Tier::Host, Tier::Nvme, 3e9, 80);
  t.Set(Tier::Object, Tier::Nvme, 1e9, 20000);
  t.Set(Tier::Object, Tier::Host, 1e9, 20000);
  return t;
}

namespace {
using Clock = std::chrono::steady_clock;

template <typename F>
double TimeBytesPerSec(F&& fn, size_t bytes, int iters) {
  fn();  // warm
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) fn();
  double s = std::chrono::duration<double>(Clock::now() - t0).count();
  return s > 0 ? double(bytes) * iters / s : 0;
}
}  // namespace

Result<BandwidthTable> ProbeBandwidth(hal::Device& dev, const ProbeOptions& opts) {
  BandwidthTable t;
  const size_t n = opts.buffer_bytes;
  auto host = dev.MakePinnedBuffer(n);
  auto host2 = Buffer::AllocateHost(n);
  auto dev1 = dev.MakeDeviceBuffer(n);
  auto dev2 = dev.MakeDeviceBuffer(n);
  if (!host || !dev1 || !dev2) return Status::OutOfMemory("bandwidth probe buffers");
  std::memset(host->data(), 1, n);
  ASTER_ASSIGN_OR_RETURN(auto stream, dev.CreateStream());

  double h2d = TimeBytesPerSec([&] { dev.CopyHostToDevice(dev1->data(), host->data(), n, stream); dev.Synchronize(stream); }, n, opts.iterations);
  double d2h = TimeBytesPerSec([&] { dev.CopyDeviceToHost(host->data(), dev1->data(), n, stream); dev.Synchronize(stream); }, n, opts.iterations);
  double d2d = TimeBytesPerSec([&] { dev.CopyDeviceToDevice(dev2->data(), dev1->data(), n, stream); dev.Synchronize(stream); }, n, opts.iterations);
  double h2h = TimeBytesPerSec([&] { std::memcpy(host2->data(), host->data(), n); }, n, opts.iterations);
  t.Set(Tier::Host, Tier::Hbm, h2d, 10);
  t.Set(Tier::Hbm, Tier::Host, d2h, 10);
  t.Set(Tier::Hbm, Tier::Hbm, d2d * 2, 0);  // copy reads and writes
  t.Set(Tier::Host, Tier::Host, h2h * 2, 0);

  if (!opts.nvme_probe_path.empty()) {
    std::string p = opts.nvme_probe_path;
    Status w = WriteFile(p, host->data(), n, true);
    if (w.ok()) {
      auto reader = MakeFileReader(nullptr, false);
      PosixFileReader pr(nullptr);
      double nvme_h = TimeBytesPerSec([&] { pr.ReadToHost(p, 0, static_cast<uint32_t>(n), host2->data()); }, n, opts.iterations);
      t.Set(Tier::Nvme, Tier::Host, nvme_h, 80);
      t.Set(Tier::Host, Tier::Nvme, nvme_h / 2, 80);
      if (opts.probe_gds) {
        auto gds = MakeFileReader(nullptr, true);
        double nvme_d = TimeBytesPerSec([&] { gds->ReadToDevice(p, 0, static_cast<uint32_t>(n), dev1->data(), stream); dev.Synchronize(stream); }, n, opts.iterations);
        t.Set(Tier::Nvme, Tier::Hbm, nvme_d, 100);
      } else {
        t.Set(Tier::Nvme, Tier::Hbm, std::min(nvme_h, h2d), 100);
      }
      std::remove(p.c_str());
    } else {
      ASTER_LOG(Warn, "nvme probe skipped: %s", w.ToString().c_str());
    }
  }
  dev.DestroyStream(stream);
  ASTER_LOG(Info, "bandwidth probe:\n%s", t.ToString().c_str());
  return t;
}

Result<BandwidthTable> LoadOrProbe(hal::Device& dev, const ProbeOptions& opts, const std::string& cache_path) {
  BandwidthTable t;
  if (!cache_path.empty() && t.Load(cache_path).ok()) return t;
  ASTER_ASSIGN_OR_RETURN(t, ProbeBandwidth(dev, opts));
  if (!cache_path.empty()) {
    Status s = t.Save(cache_path);
    if (!s.ok()) ASTER_LOG(Warn, "could not cache bandwidth table: %s", s.ToString().c_str());
  }
  return t;
}

}  // namespace aster::memory
