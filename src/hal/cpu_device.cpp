#include "aster/hal/cpu_device.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace aster::hal {

namespace {
struct CpuEvent {
  std::chrono::steady_clock::time_point t;
};
}  // namespace

CpuDeviceImpl::CpuDeviceImpl() {
  info_.backend = Backend::Cpu;
  info_.device_id = -1;
  info_.name = "cpu";
  info_.total_memory = size_t(64) << 30;
  info_.free_memory = info_.total_memory;
  info_.coherent_host_access = true;
  info_.sm_count = static_cast<int>(std::thread::hardware_concurrency());
  info_.shared_mem_per_block = 48 * 1024;
}

Result<void*> CpuDeviceImpl::AllocateDevice(size_t bytes, Stream) {
  void* p = std::calloc(bytes ? bytes : 1, 1);
  if (!p) return Status::OutOfMemory("host alloc");
  return p;
}
void CpuDeviceImpl::FreeDevice(void* p, size_t, Stream) { std::free(p); }
Result<void*> CpuDeviceImpl::AllocatePinned(size_t bytes) { return AllocateDevice(bytes, {}); }
void CpuDeviceImpl::FreePinned(void* p, size_t) { std::free(p); }
Result<void*> CpuDeviceImpl::AllocateManaged(size_t bytes) { return AllocateDevice(bytes, {}); }
void CpuDeviceImpl::FreeManaged(void* p, size_t) { std::free(p); }

Result<Stream> CpuDeviceImpl::CreateStream() { return Stream{nullptr, -1}; }
void CpuDeviceImpl::DestroyStream(Stream) {}
Status CpuDeviceImpl::Synchronize(Stream) { return Status::OK(); }
Status CpuDeviceImpl::SynchronizeDevice() { return Status::OK(); }
Result<Event> CpuDeviceImpl::RecordEvent(Stream) {
  return Event{new CpuEvent{std::chrono::steady_clock::now()}};
}
Status CpuDeviceImpl::WaitEvent(Stream, Event) { return Status::OK(); }
Result<float> CpuDeviceImpl::ElapsedMs(Event a, Event b) {
  auto* ea = static_cast<CpuEvent*>(a.handle);
  auto* eb = static_cast<CpuEvent*>(b.handle);
  return std::chrono::duration<float, std::milli>(eb->t - ea->t).count();
}
void CpuDeviceImpl::DestroyEvent(Event e) { delete static_cast<CpuEvent*>(e.handle); }

Status CpuDeviceImpl::CopyHostToDevice(void* dst, const void* src, size_t bytes, Stream) {
  std::memcpy(dst, src, bytes);
  return Status::OK();
}
Status CpuDeviceImpl::CopyDeviceToHost(void* dst, const void* src, size_t bytes, Stream) {
  std::memcpy(dst, src, bytes);
  return Status::OK();
}
Status CpuDeviceImpl::CopyDeviceToDevice(void* dst, const void* src, size_t bytes, Stream) {
  std::memmove(dst, src, bytes);
  return Status::OK();
}
Status CpuDeviceImpl::Memset(void* dst, int value, size_t bytes, Stream) {
  std::memset(dst, value, bytes);
  return Status::OK();
}
Status CpuDeviceImpl::PrefetchManaged(const void*, size_t, Stream) { return Status::OK(); }
Status CpuDeviceImpl::AdviseAccess(const void*, size_t, bool) { return Status::OK(); }

}  // namespace aster::hal
