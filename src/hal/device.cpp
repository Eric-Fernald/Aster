#include "aster/hal/device.hpp"

#include "aster/hal/cpu_device.hpp"
#include "aster/hal/cuda_device.hpp"

#if ASTER_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace aster::hal {

namespace {
struct DevCtx {
  Device* dev;
  MemorySpace space;
};
void DeviceDeleter(void* p, size_t bytes, void* ctx) {
  auto* c = static_cast<DevCtx*>(ctx);
  switch (c->space) {
    case MemorySpace::Device: c->dev->FreeDevice(p, bytes); break;
    case MemorySpace::HostPinned: c->dev->FreePinned(p, bytes); break;
    case MemorySpace::Managed: c->dev->FreeManaged(p, bytes); break;
    default: break;
  }
  delete c;
}
std::shared_ptr<Buffer> Wrap(Device* d, void* p, size_t bytes, MemorySpace space) {
  return std::make_shared<Buffer>(p, bytes, space, DeviceDeleter, new DevCtx{d, space});
}
}  // namespace

std::shared_ptr<Buffer> Device::MakeDeviceBuffer(size_t bytes, Stream s) {
  auto r = AllocateDevice(bytes, s);
  if (!r.ok()) return nullptr;
  return Wrap(this, r.value(), bytes, MemorySpace::Device);
}

std::shared_ptr<Buffer> Device::MakePinnedBuffer(size_t bytes) {
  auto r = AllocatePinned(bytes);
  if (!r.ok()) return nullptr;
  return Wrap(this, r.value(), bytes, MemorySpace::HostPinned);
}

std::shared_ptr<Buffer> Device::MakeManagedBuffer(size_t bytes) {
  auto r = AllocateManaged(bytes);
  if (!r.ok()) return nullptr;
  return Wrap(this, r.value(), bytes, MemorySpace::Managed);
}

Result<std::shared_ptr<Buffer>> Device::ToDevice(const Buffer& host, Stream s) {
  auto b = MakeDeviceBuffer(host.size(), s);
  if (!b) return Status::OutOfMemory("device buffer of " + std::to_string(host.size()) + " bytes");
  ASTER_RETURN_NOT_OK(CopyHostToDevice(b->data(), host.data(), host.size(), s));
  return b;
}

Result<std::shared_ptr<Buffer>> Device::ToHost(const Buffer& dev, Stream s) {
  auto b = MakePinnedBuffer(dev.size());
  if (!b) b = Buffer::AllocateHost(dev.size());
  ASTER_RETURN_NOT_OK(CopyDeviceToHost(b->data(), dev.data(), dev.size(), s));
  return b;
}

Result<Column> Device::ColumnToDevice(const Column& c, Stream s) {
  Column out = c;
  if (c.validity && c.validity->space() != MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.validity, ToDevice(*c.validity, s)); }
  if (c.offsets && c.offsets->space() != MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.offsets, ToDevice(*c.offsets, s)); }
  if (c.values && c.values->space() != MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.values, ToDevice(*c.values, s)); }
  if (c.dictionary && c.dictionary->space() != MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.dictionary, ToDevice(*c.dictionary, s)); }
  if (c.dictionary_offsets && c.dictionary_offsets->space() != MemorySpace::Device) {
    ASTER_ASSIGN_OR_RETURN(out.dictionary_offsets, ToDevice(*c.dictionary_offsets, s));
  }
  return out;
}

Result<Column> Device::ColumnToHost(const Column& c, Stream s) {
  Column out = c;
  if (c.validity && c.validity->space() == MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.validity, ToHost(*c.validity, s)); }
  if (c.offsets && c.offsets->space() == MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.offsets, ToHost(*c.offsets, s)); }
  if (c.values && c.values->space() == MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.values, ToHost(*c.values, s)); }
  if (c.dictionary && c.dictionary->space() == MemorySpace::Device) { ASTER_ASSIGN_OR_RETURN(out.dictionary, ToHost(*c.dictionary, s)); }
  if (c.dictionary_offsets && c.dictionary_offsets->space() == MemorySpace::Device) {
    ASTER_ASSIGN_OR_RETURN(out.dictionary_offsets, ToHost(*c.dictionary_offsets, s));
  }
  return out;
}

const char* BackendName(Backend b) {
  switch (b) {
    case Backend::Cuda: return "cuda";
    case Backend::Rocm: return "rocm";
    case Backend::OneApi: return "oneapi";
    default: return "cpu";
  }
}

DevicePtr CpuDevice() {
  static DevicePtr dev = std::make_shared<CpuDeviceImpl>();
  return dev;
}

std::vector<DeviceInfo> EnumerateDevices() {
  std::vector<DeviceInfo> out;
  out.push_back(CpuDevice()->info());
#if ASTER_HAVE_CUDA
  int n = 0;
  if (cudaGetDeviceCount(&n) == cudaSuccess) {
    for (int i = 0; i < n; ++i) {
      auto d = CudaDeviceImpl::Open(i);
      if (d.ok()) out.push_back(d.value()->info());
    }
  }
#endif
  return out;
}

Result<DevicePtr> OpenDevice(Backend backend, int device_id) {
  switch (backend) {
    case Backend::Cpu: return CpuDevice();
#if ASTER_HAVE_CUDA
    case Backend::Cuda: {
      ASTER_ASSIGN_OR_RETURN(auto d, CudaDeviceImpl::Open(device_id));
      return DevicePtr(d);
    }
#endif
    default:
      return Status::NotSupported(std::string("backend ") + BackendName(backend) + " not compiled in");
  }
}

}  // namespace aster::hal
