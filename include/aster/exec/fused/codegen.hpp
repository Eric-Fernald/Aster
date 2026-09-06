#pragma once
#include <string>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/planner/physical_plan.hpp"
#include "aster/storage/encoding.hpp"

namespace aster::exec::fused {

struct ColumnBinding {
  int index = 0;
  DataType type;
  storage::Encoding encoding = storage::Encoding::Plain;
};

struct KernelSpec {
  std::string name;               // kernel symbol
  std::string source;             // CUDA C++ generated for NVRTC
  std::string shape_hash;
  std::vector<ColumnBinding> inputs;
  uint32_t tile_rows = 32 * 1024;
  uint32_t block_size = 256;
  size_t shared_bytes = 0;
  bool has_partial_aggregate = false;
  bool has_hash_probe = false;
  uint32_t num_outputs = 0;
};

// Generates one kernel per pipeline shape. Each operator becomes a device function invoked in
// sequence inside the tile loop; intermediates live in registers and shared memory. Compressed
// columns are decoded lazily and only for rows that survived every earlier filter.
class Codegen {
 public:
  static Result<KernelSpec> Generate(const planner::Pipeline& p, const std::vector<ColumnBinding>& inputs,
                                     uint32_t tile_rows);
  static std::string CudaType(const DataType& t);
  static Result<std::string> ExprToCuda(const plan::Expr& e, const std::vector<ColumnBinding>& inputs,
                                        const std::string& row_var);
  static std::string DecodeSnippet(const ColumnBinding& b, const std::string& row_var);
  static std::string Preamble();
};

}  // namespace aster::exec::fused
