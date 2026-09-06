#pragma once
#include <memory>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "dlpack/dlpack.h"

namespace aster::interop {

// Zero copy handoff of result columns to PyTorch, JAX and cuDF. The managed tensor keeps the
// column's buffers alive until the consumer calls the deleter.
class DlpackExporter {
 public:
  static Result<DLManagedTensor*> Export(const Column& col, int device_id = 0);
  // Column-major 2D tensor for a fixed width numeric batch; rows x columns after a transpose copy.
  static Result<DLManagedTensor*> ExportMatrix(const RecordBatch& batch, int device_id = 0);
  static Result<DLDataType> DataTypeOf(const DataType& t);
  static Result<Column> Import(DLManagedTensor* tensor);  // wraps foreign memory, no copy
};

}  // namespace aster::interop
