#include "aster/hal/cuda_device.hpp"

#if ASTER_HAVE_CUDA
#include <cuda_runtime.h>

namespace aster::hal {

namespace {
Status Check(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return Status::OK();
  if (e == cudaErrorMemoryAllocation) return Status::OutOfMemory(what);
  return Status::Internal(std::string(what) + ": " + cudaGetErrorString(e));
}
inline cudaStream_t S(Stream s) { return static_cast<cudaStream_t>(s.handle); }
inline cudaEvent_t E(Event e) { return static_cast<cudaEvent_t>(e.handle); }
}  // namespace

Result<std::shared_ptr<CudaDeviceImpl>> CudaDeviceImpl::Open(int device_id) {
  std::shared_ptr<CudaDeviceImpl> d(new CudaDeviceImpl(device_id));
  ASTER_RETURN_NOT_OK(Check(cudaSetDevice(device_id), "cudaSetDevice"));
  cudaDeviceProp prop{};
  ASTER_RETURN_NOT_OK(Check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties"));
  d->info_.backend = Backend::Cuda;
  d->info_.name = prop.name;
  d->info_.total_memory = prop.totalGlobalMem;
  d->info_.sm_count = prop.multiProcessorCount;
  d->info_.shared_mem_per_block = prop.sharedMemPerBlockOptin;
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrPageableMemoryAccess, device_id);
  d->info_.coherent_host_access = attr != 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMASupported, device_id);
  d->info_.gds_capable = attr != 0;
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  d->info_.free_memory = free_b;
  return d;
}

Status CudaDeviceImpl::Activate() const { return Check(cudaSetDevice(info_.device_id), "cudaSetDevice"); }

Result<void*> CudaDeviceImpl::AllocateDevice(size_t bytes, Stream s) {
  ASTER_RETURN_NOT_OK(Activate());
  void* p = nullptr;
  ASTER_RETURN_NOT_OK(Check(cudaMallocAsync(&p, bytes ? bytes : 1, S(s)), "cudaMallocAsync"));
  return p;
}
void CudaDeviceImpl::FreeDevice(void* p, size_t, Stream s) { cudaFreeAsync(p, S(s)); }

Result<void*> CudaDeviceImpl::AllocatePinned(size_t bytes) {
  void* p = nullptr;
  ASTER_RETURN_NOT_OK(Check(cudaHostAlloc(&p, bytes ? bytes : 1, cudaHostAllocPortable), "cudaHostAlloc"));
  return p;
}
void CudaDeviceImpl::FreePinned(void* p, size_t) { cudaFreeHost(p); }

Result<void*> CudaDeviceImpl::AllocateManaged(size_t bytes) {
  void* p = nullptr;
  ASTER_RETURN_NOT_OK(Check(cudaMallocManaged(&p, bytes ? bytes : 1), "cudaMallocManaged"));
  return p;
}
void CudaDeviceImpl::FreeManaged(void* p, size_t) { cudaFree(p); }

Result<Stream> CudaDeviceImpl::CreateStream() {
  ASTER_RETURN_NOT_OK(Activate());
  cudaStream_t s;
  ASTER_RETURN_NOT_OK(Check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate"));
  return Stream{s, info_.device_id};
}
void CudaDeviceImpl::DestroyStream(Stream s) { if (s.handle) cudaStreamDestroy(S(s)); }
Status CudaDeviceImpl::Synchronize(Stream s) { return Check(cudaStreamSynchronize(S(s)), "cudaStreamSynchronize"); }
Status CudaDeviceImpl::SynchronizeDevice() {
  ASTER_RETURN_NOT_OK(Activate());
  return Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
Result<Event> CudaDeviceImpl::RecordEvent(Stream s) {
  cudaEvent_t e;
  ASTER_RETURN_NOT_OK(Check(cudaEventCreate(&e), "cudaEventCreate"));
  ASTER_RETURN_NOT_OK(Check(cudaEventRecord(e, S(s)), "cudaEventRecord"));
  return Event{e};
}
Status CudaDeviceImpl::WaitEvent(Stream s, Event e) { return Check(cudaStreamWaitEvent(S(s), E(e), 0), "cudaStreamWaitEvent"); }
Result<float> CudaDeviceImpl::ElapsedMs(Event a, Event b) {
  ASTER_RETURN_NOT_OK(Check(cudaEventSynchronize(E(b)), "cudaEventSynchronize"));
  float ms = 0;
  ASTER_RETURN_NOT_OK(Check(cudaEventElapsedTime(&ms, E(a), E(b)), "cudaEventElapsedTime"));
  return ms;
}
void CudaDeviceImpl::DestroyEvent(Event e) { if (e.handle) cudaEventDestroy(E(e)); }

Status CudaDeviceImpl::CopyHostToDevice(void* dst, const void* src, size_t bytes, Stream s) {
  return Check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, S(s)), "H2D");
}
Status CudaDeviceImpl::CopyDeviceToHost(void* dst, const void* src, size_t bytes, Stream s) {
  return Check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, S(s)), "D2H");
}
Status CudaDeviceImpl::CopyDeviceToDevice(void* dst, const void* src, size_t bytes, Stream s) {
  return Check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, S(s)), "D2D");
}
Status CudaDeviceImpl::Memset(void* dst, int value, size_t bytes, Stream s) {
  return Check(cudaMemsetAsync(dst, value, bytes, S(s)), "cudaMemsetAsync");
}
Status CudaDeviceImpl::PrefetchManaged(const void* ptr, size_t bytes, Stream s) {
  return Check(cudaMemPrefetchAsync(ptr, bytes, info_.device_id, S(s)), "cudaMemPrefetchAsync");
}
Status CudaDeviceImpl::AdviseAccess(const void* ptr, size_t bytes, bool preferred_device) {
  int loc = preferred_device ? info_.device_id : cudaCpuDeviceId;
  ASTER_RETURN_NOT_OK(Check(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc), "cudaMemAdvise"));
  return Check(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetAccessedBy, info_.device_id), "cudaMemAdvise");
}

}  // namespace aster::hal
#endif
