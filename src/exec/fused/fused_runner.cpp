#include "aster/exec/fused/fused_runner.hpp"

#include <chrono>

#include "aster/common/log.hpp"

#if ASTER_HAVE_NVRTC
#include <cuda.h>
#endif

namespace aster::exec::fused {

FusedPipelineRunner::FusedPipelineRunner(JitCompiler& jit, const planner::Pipeline& pipeline, std::vector<ColumnBinding> inputs, uint32_t tile_rows)
    : jit_(jit), pipeline_(pipeline), inputs_(std::move(inputs)), tile_rows_(tile_rows) {}

bool FusedPipelineRunner::CanRun(const planner::Pipeline& p, hal::Backend backend) {
  return p.fusable && backend == hal::Backend::Cuda && JitCompiler::Available();
}

Status FusedPipelineRunner::Prepare() {
  ASTER_ASSIGN_OR_RETURN(spec_, Codegen::Generate(pipeline_, inputs_, tile_rows_));
  ASTER_ASSIGN_OR_RETURN(kernel_, jit_.GetOrCompile(spec_));
  return Status::OK();
}

Result<RecordBatchPtr> FusedPipelineRunner::RunTile(const std::vector<memory::PageHandle>& pages, uint32_t num_rows, ExecContext& ctx) {
#if ASTER_HAVE_NVRTC
  if (!kernel_ || !kernel_->function) return Status::Invalid("kernel not prepared");
  struct PageIn { const void* data; uint32_t bytes; };
  std::vector<PageIn> host_pages;
  for (const auto& p : pages) host_pages.push_back({p.data(), p.bytes()});
  hal::Device& dev = *ctx.device;
  auto d_pages = dev.MakeDeviceBuffer(host_pages.size() * sizeof(PageIn), ctx.stream);
  ASTER_RETURN_NOT_OK(dev.CopyHostToDevice(d_pages->data(), host_pages.data(), host_pages.size() * sizeof(PageIn), ctx.stream));

  // Output: row indices plus projected columns, one buffer each, plus a device counter.
  uint32_t nout = spec_.num_outputs ? spec_.num_outputs : 1;
  std::vector<std::shared_ptr<Buffer>> outs;
  std::vector<void*> out_ptrs;
  for (uint32_t i = 0; i < nout; ++i) { outs.push_back(dev.MakeDeviceBuffer(size_t(num_rows) * 8, ctx.stream)); out_ptrs.push_back(outs.back()->data()); }
  auto d_out_ptrs = dev.MakeDeviceBuffer(out_ptrs.size() * sizeof(void*), ctx.stream);
  ASTER_RETURN_NOT_OK(dev.CopyHostToDevice(d_out_ptrs->data(), out_ptrs.data(), out_ptrs.size() * sizeof(void*), ctx.stream));
  auto d_count = dev.MakeDeviceBuffer(sizeof(uint32_t), ctx.stream);
  ASTER_RETURN_NOT_OK(dev.Memset(d_count->data(), 0, sizeof(uint32_t), ctx.stream));
  struct Outputs { void** cols; uint32_t* count; } outputs{reinterpret_cast<void**>(d_out_ptrs->data()), reinterpret_cast<uint32_t*>(d_count->data())};

  uint32_t table_slots = 1u << 16;
  auto d_keys = dev.MakeDeviceBuffer(size_t(table_slots) * 8, ctx.stream);
  auto d_vals = dev.MakeDeviceBuffer(size_t(table_slots) * 8 * 8, ctx.stream);
  auto d_counts = dev.MakeDeviceBuffer(size_t(table_slots) * 4 * 8, ctx.stream);
  ASTER_RETURN_NOT_OK(dev.Memset(d_keys->data(), 0, d_keys->size(), ctx.stream));
  ASTER_RETURN_NOT_OK(dev.Memset(d_vals->data(), 0, d_vals->size(), ctx.stream));
  ASTER_RETURN_NOT_OK(dev.Memset(d_counts->data(), 0, d_counts->size(), ctx.stream));

  void* pages_arg = d_pages->data();
  void* keys_arg = d_keys->data(); void* vals_arg = d_vals->data(); void* counts_arg = d_counts->data();
  void* args[] = {&pages_arg, &num_rows, &outputs, &keys_arg, &vals_arg, &counts_arg, &table_slots};
  unsigned grid = (num_rows + spec_.block_size - 1) / spec_.block_size;
  auto t0 = std::chrono::steady_clock::now();
  CUresult r = cuLaunchKernel(static_cast<CUfunction>(kernel_->function), grid, 1, 1, spec_.block_size, 1, 1,
                              static_cast<unsigned>(spec_.shared_bytes), static_cast<CUstream>(ctx.stream.handle), args, nullptr);
  if (r != CUDA_SUCCESS) return Status::Internal("cuLaunchKernel failed");
  ASTER_RETURN_NOT_OK(dev.Synchronize(ctx.stream));
  stats_.kernel_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  stats_.launches++;
  stats_.rows_in += num_rows;

  uint32_t count = 0;
  ASTER_RETURN_NOT_OK(dev.CopyDeviceToHost(&count, d_count->data(), sizeof count, ctx.stream));
  ASTER_RETURN_NOT_OK(dev.Synchronize(ctx.stream));
  stats_.rows_out += count;
  auto batch = std::make_shared<RecordBatch>();
  batch->schema.fields.push_back({"row", DataType::Of(TypeId::UInt32), false});
  Column rows;
  rows.type = DataType::Of(TypeId::UInt32);
  rows.length = count;
  rows.values = outs[0];
  batch->columns.push_back(std::move(rows));
  const plan::Rel& sink = *pipeline_.sink();
  for (uint32_t i = 1; i < nout; ++i) {
    Column c;
    c.type = sink.output.fields[i - 1].type;
    c.length = count;
    c.values = outs[i];
    batch->schema.fields.push_back(sink.output.fields[i - 1]);
    batch->columns.push_back(std::move(c));
  }
  return batch;
#else
  (void)pages; (void)num_rows; (void)ctx;
  return Status::NotSupported("fused kernels need NVRTC");
#endif
}

}  // namespace aster::exec::fused
