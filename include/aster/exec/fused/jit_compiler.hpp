#pragma once
#include <memory>
#include <string>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/exec/fused/codegen.hpp"
#include "aster/exec/fused/kernel_cache.hpp"

namespace aster::exec::fused {

struct JitOptions {
  int sm_major = 8, sm_minor = 0;
  bool debug = false;
  bool emit_ptx = false;   // false = cubin via nvJitLink or --cubin when available
  std::vector<std::string> extra_flags;
};

// NVRTC front end. Without CUDA it still validates generated source structurally so the codegen
// path stays testable on CPU only machines.
class JitCompiler {
 public:
  JitCompiler(KernelCache& cache, JitOptions opts = {});
  static bool Available();
  Result<std::shared_ptr<CompiledKernel>> GetOrCompile(const KernelSpec& spec);
  Result<std::shared_ptr<CompiledKernel>> Compile(const KernelSpec& spec);
  Status Load(CompiledKernel& k);   // module load into the current context
  const JitOptions& options() const { return opts_; }

 private:
  KernelCache& cache_;
  JitOptions opts_;
};

}  // namespace aster::exec::fused
