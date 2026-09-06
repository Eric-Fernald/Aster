#pragma once
#include "aster/hal/device.hpp"

namespace aster::hal {

#if ASTER_HAVE_CUDA
class CudaDeviceImpl final : public Device {
 public:
  static Result<std::shared_ptr<CudaDeviceImpl>> Open(int device_id);
  const DeviceInfo& info() const override { return info_; }
  Backend backend() const override { return Backend::Cuda; }

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
  explicit CudaDeviceImpl(int id) { info_.device_id = id; }
  Status Activate() const;
  DeviceInfo info_;
};
#endif

}  // namespace aster::hal
