#include "aster/exec/fused/jit_compiler.hpp"

#include <chrono>

#include "aster/common/log.hpp"

#if ASTER_HAVE_NVRTC
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace aster::exec::fused {

JitCompiler::JitCompiler(KernelCache& cache, JitOptions opts) : cache_(cache), opts_(std::move(opts)) {}

bool JitCompiler::Available() {
#if ASTER_HAVE_NVRTC
  return true;
#else
  return false;
#endif
}

Result<std::shared_ptr<CompiledKernel>> JitCompiler::GetOrCompile(const KernelSpec& spec) {
  if (auto k = cache_.Get(spec.shape_hash)) return k;
  auto disk = cache_.LoadImage(spec.shape_hash);
  if (disk.ok()) {
    auto k = std::make_shared<CompiledKernel>();
    k->shape_hash = spec.shape_hash;
    k->name = spec.name;
    k->image = std::move(disk.value());
    k->from_disk = true;
    Status s = Load(*k);
    if (s.ok()) { cache_.Put(k); return k; }
    ASTER_LOG(Warn, "cached kernel %s failed to load: %s", spec.shape_hash.c_str(), s.ToString().c_str());
  }
  ASTER_ASSIGN_OR_RETURN(auto k, Compile(spec));
  cache_.Put(k);
  Status s = cache_.Persist(*k);
  if (!s.ok()) ASTER_LOG(Debug, "kernel persist skipped: %s", s.ToString().c_str());
  return k;
}

Result<std::shared_ptr<CompiledKernel>> JitCompiler::Compile(const KernelSpec& spec) {
  auto t0 = std::chrono::steady_clock::now();
  auto k = std::make_shared<CompiledKernel>();
  k->shape_hash = spec.shape_hash;
  k->name = spec.name;
#if ASTER_HAVE_NVRTC
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, spec.source.c_str(), (spec.name + ".cu").c_str(), 0, nullptr, nullptr) != NVRTC_SUCCESS)
    return Status::Internal("nvrtcCreateProgram failed");
  std::string arch = "--gpu-architecture=sm_" + std::to_string(opts_.sm_major) + std::to_string(opts_.sm_minor);
  std::vector<std::string> flags = {arch, "--std=c++17", "-default-device", "--extra-device-vectorization"};
  if (opts_.debug) flags.push_back("-G");
  for (const auto& f : opts_.extra_flags) flags.push_back(f);
  std::vector<const char*> cflags;
  for (const auto& f : flags) cflags.push_back(f.c_str());
  nvrtcResult r = nvrtcCompileProgram(prog, static_cast<int>(cflags.size()), cflags.data());
  size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  if (log_size) nvrtcGetProgramLog(prog, log.data());
  if (r != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return Status::Internal("nvrtc: " + log); }
  size_t n = 0;
  if (opts_.emit_ptx) { nvrtcGetPTXSize(prog, &n); k->image.resize(n); nvrtcGetPTX(prog, k->image.data()); }
  else { nvrtcGetCUBINSize(prog, &n); k->image.resize(n); nvrtcGetCUBIN(prog, k->image.data()); }
  nvrtcDestroyProgram(&prog);
  ASTER_RETURN_NOT_OK(Load(*k));
#else
  // No compiler: keep the generated source as the image so tooling can inspect it.
  if (spec.source.find("__global__") == std::string::npos) return Status::Internal("generated kernel has no entry point");
  k->image.assign(spec.source.begin(), spec.source.end());
#endif
  k->compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ASTER_LOG(Info, "jit compiled %s in %.1f ms (%zu bytes)", spec.name.c_str(), k->compile_ms, k->image.size());
  return k;
}

Status JitCompiler::Load(CompiledKernel& k) {
#if ASTER_HAVE_NVRTC
  CUmodule mod;
  if (cuModuleLoadData(&mod, k.image.data()) != CUDA_SUCCESS) return Status::Internal("cuModuleLoadData failed");
  CUfunction fn;
  if (cuModuleGetFunction(&fn, mod, k.name.c_str()) != CUDA_SUCCESS) { cuModuleUnload(mod); return Status::Internal("kernel symbol not found: " + k.name); }
  k.module = mod;
  k.function = fn;
  return Status::OK();
#else
  (void)k;
  return Status::OK();
#endif
}

}  // namespace aster::exec::fused
