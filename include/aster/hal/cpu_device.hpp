#pragma once
#include "aster/hal/device.hpp"

namespace aster::hal {

// Host only backend. Device memory is host memory; copies are memcpy. Used for fallback and tests.
class CpuDeviceImpl final : public Device {
 public:
  CpuDeviceImpl();
  const DeviceInfo& info() const override { return info_; }
  Backend backend() const override { return Backend::Cpu; }

  Result<void*> AllocateDevice(size_t bytes, Stream s) override;
  void FreeDevice(void* p, size_t bytes, Stream s) override;
  Result<void*> AllocatePinned(size_t bytes) override;
  void FreePinned(void* p, size_t bytes) override;
  Result<void*> AllocateManaged(size_t bytes) override;
  void FreeManaged(void* p, size_t bytes) override;

  Result<Stream> CreateStream() override;
  void DestroyStream(Stream s) override;
  Status Synchronize(Stream s) override;
  Status SynchronizeDevice() override;
  Result<Event> RecordEvent(Stream s) override;
  Status WaitEvent(Stream s, Event e) override;
  Result<float> ElapsedMs(Event a, Event b) override;
  void DestroyEvent(Event e) override;

  Status CopyHostToDevice(void* dst, const void* src, size_t bytes, Stream s) override;
  Status CopyDeviceToHost(void* dst, const void* src, size_t bytes, Stream s) override;
  Status CopyDeviceToDevice(void* dst, const void* src, size_t bytes, Stream s) override;
  Status Memset(void* dst, int value, size_t bytes, Stream s) override;
  Status PrefetchManaged(const void* ptr, size_t bytes, Stream s) override;
  Status AdviseAccess(const void* ptr, size_t bytes, bool preferred_device) override;

 private:
  DeviceInfo info_;
};

}  // namespace aster::hal
