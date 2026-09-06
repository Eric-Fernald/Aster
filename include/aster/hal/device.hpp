#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"

namespace aster::hal {

enum class Backend { Cpu, Cuda, Rocm, OneApi };

struct DeviceInfo {
  Backend backend = Backend::Cpu;
  int device_id = -1;
  std::string name = "cpu";
  size_t total_memory = 0;
  size_t free_memory = 0;
  bool coherent_host_access = false;  // HMM / ATS available
  bool gds_capable = false;
  int sm_count = 0;
  size_t shared_mem_per_block = 0;
};

struct Stream {
  void* handle = nullptr;
  int device_id = -1;
};

struct Event {
  void* handle = nullptr;
};

// Uniform device interface. All backends implement copies, allocation, streams and events.
class Device {
 public:
  virtual ~Device() = default;
  virtual const DeviceInfo& info() const = 0;
  virtual Backend backend() const = 0;

  virtual Result<void*> AllocateDevice(size_t bytes, Stream s = {}) = 0;
  virtual void FreeDevice(void* p, size_t bytes, Stream s = {}) = 0;
  virtual Result<void*> AllocatePinned(size_t bytes) = 0;
  virtual void FreePinned(void* p, size_t bytes) = 0;
  virtual Result<void*> AllocateManaged(size_t bytes) = 0;
  virtual void FreeManaged(void* p, size_t bytes) = 0;

  virtual Result<Stream> CreateStream() = 0;
  virtual void DestroyStream(Stream s) = 0;
  virtual Status Synchronize(Stream s) = 0;
  virtual Status SynchronizeDevice() = 0;
  virtual Result<Event> RecordEvent(Stream s) = 0;
  virtual Status WaitEvent(Stream s, Event e) = 0;
  virtual Result<float> ElapsedMs(Event a, Event b) = 0;
  virtual void DestroyEvent(Event e) = 0;

  virtual Status CopyHostToDevice(void* dst, const void* src, size_t bytes, Stream s = {}) = 0;
  virtual Status CopyDeviceToHost(void* dst, const void* src, size_t bytes, Stream s = {}) = 0;
  virtual Status CopyDeviceToDevice(void* dst, const void* src, size_t bytes, Stream s = {}) = 0;
  virtual Status Memset(void* dst, int value, size_t bytes, Stream s = {}) = 0;
  virtual Status PrefetchManaged(const void* ptr, size_t bytes, Stream s = {}) = 0;
  virtual Status AdviseAccess(const void* ptr, size_t bytes, bool preferred_device) = 0;

  std::shared_ptr<Buffer> MakeDeviceBuffer(size_t bytes, Stream s = {});
  std::shared_ptr<Buffer> MakePinnedBuffer(size_t bytes);
  std::shared_ptr<Buffer> MakeManagedBuffer(size_t bytes);
  Result<std::shared_ptr<Buffer>> ToDevice(const Buffer& host, Stream s = {});
  Result<std::shared_ptr<Buffer>> ToHost(const Buffer& dev, Stream s = {});
  Result<Column> ColumnToDevice(const Column& c, Stream s = {});
  Result<Column> ColumnToHost(const Column& c, Stream s = {});
};

using DevicePtr = std::shared_ptr<Device>;

std::vector<DeviceInfo> EnumerateDevices();
Result<DevicePtr> OpenDevice(Backend backend, int device_id);
DevicePtr CpuDevice();
const char* BackendName(Backend b);

}  // namespace aster::hal
