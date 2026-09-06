#include "aster/interop/dlpack_export.hpp"

#include <cstring>

namespace aster::interop {

namespace {
struct Holder {
  std::shared_ptr<Buffer> buf;
  std::vector<std::shared_ptr<Buffer>> extra;
  int64_t shape[2];
  int64_t strides[2];
};
void Deleter(DLManagedTensor* t) {
  delete static_cast<Holder*>(t->manager_ctx);
  delete t;
}
DLDevice DeviceOf(MemorySpace s, int device_id) {
  switch (s) {
    case MemorySpace::Device: return {kDLCUDA, device_id};
    case MemorySpace::HostPinned: return {kDLCUDAHost, device_id};
    case MemorySpace::Managed: return {kDLCUDAManaged, device_id};
    default: return {kDLCPU, 0};
  }
}
}  // namespace

Result<DLDataType> DlpackExporter::DataTypeOf(const DataType& t) {
  switch (t.id) {
    case TypeId::Bool: return DLDataType{kDLBool, 8, 1};
    case TypeId::Int8: return DLDataType{kDLInt, 8, 1};
    case TypeId::Int16: return DLDataType{kDLInt, 16, 1};
    case TypeId::Int32: case TypeId::Date32: return DLDataType{kDLInt, 32, 1};
    case TypeId::Int64: case TypeId::Timestamp: case TypeId::Decimal64: return DLDataType{kDLInt, 64, 1};
    case TypeId::UInt8: return DLDataType{kDLUInt, 8, 1};
    case TypeId::UInt16: return DLDataType{kDLUInt, 16, 1};
    case TypeId::UInt32: return DLDataType{kDLUInt, 32, 1};
    case TypeId::UInt64: return DLDataType{kDLUInt, 64, 1};
    case TypeId::Float32: case TypeId::FixedVector: return DLDataType{kDLFloat, 32, 1};
    case TypeId::Float64: return DLDataType{kDLFloat, 64, 1};
    default: return Status::NotSupported("DLPack export of " + TypeToString(t));
  }
}

Result<DLManagedTensor*> DlpackExporter::Export(const Column& col, int device_id) {
  if (col.is_dictionary_encoded()) return Status::NotSupported("decode dictionary column before DLPack export");
  if (col.null_count) return Status::NotSupported("DLPack tensors cannot carry nulls; fill nulls first");
  ASTER_ASSIGN_OR_RETURN(DLDataType dt, DataTypeOf(col.type));
  auto* h = new Holder();
  h->buf = col.values;
  auto* t = new DLManagedTensor();
  t->manager_ctx = h;
  t->deleter = Deleter;
  t->dl_tensor.data = col.values->data();
  t->dl_tensor.device = DeviceOf(col.values->space(), device_id);
  t->dl_tensor.dtype = dt;
  t->dl_tensor.byte_offset = 0;
  if (col.type.id == TypeId::FixedVector) {
    h->shape[0] = col.length; h->shape[1] = col.type.width;
    h->strides[0] = col.type.width; h->strides[1] = 1;
    t->dl_tensor.ndim = 2;
  } else {
    h->shape[0] = col.length; h->strides[0] = 1;
    t->dl_tensor.ndim = 1;
  }
  t->dl_tensor.shape = h->shape;
  t->dl_tensor.strides = h->strides;
  return t;
}

Result<DLManagedTensor*> DlpackExporter::ExportMatrix(const RecordBatch& batch, int device_id) {
  if (batch.columns.empty()) return Status::Invalid("empty batch");
  DataType t = batch.columns[0].type;
  for (const auto& c : batch.columns) {
    if (!(c.type == t) || !IsNumeric(c.type.id) && c.type.id != TypeId::Bool) return Status::NotSupported("matrix export needs uniform numeric columns");
    if (c.null_count) return Status::NotSupported("matrix export cannot carry nulls");
  }
  ASTER_ASSIGN_OR_RETURN(DLDataType dt, DataTypeOf(t));
  size_t w = TypeByteWidth(t);
  int64_t rows = batch.num_rows(), cols = static_cast<int64_t>(batch.columns.size());
  // Column major on host: each column is contiguous, so strides are (1, rows).
  auto buf = Buffer::AllocateHost(w * rows * cols);
  for (int64_t c = 0; c < cols; ++c) std::memcpy(buf->data() + c * rows * w, batch.columns[c].values->data(), rows * w);
  auto* h = new Holder();
  h->buf = buf;
  h->shape[0] = rows; h->shape[1] = cols;
  h->strides[0] = 1; h->strides[1] = rows;
  auto* tensor = new DLManagedTensor();
  tensor->manager_ctx = h;
  tensor->deleter = Deleter;
  tensor->dl_tensor = {buf->data(), DeviceOf(MemorySpace::Host, device_id), 2, dt, h->shape, h->strides, 0};
  return tensor;
}

Result<Column> DlpackExporter::Import(DLManagedTensor* t) {
  if (!t) return Status::Invalid("null tensor");
  const DLTensor& d = t->dl_tensor;
  DataType type;
  if (d.dtype.code == kDLFloat && d.dtype.bits == 32) type = d.ndim == 2 ? DataType::Vector(static_cast<uint32_t>(d.shape[1])) : DataType::Of(TypeId::Float32);
  else if (d.dtype.code == kDLFloat && d.dtype.bits == 64) type = DataType::Of(TypeId::Float64);
  else if (d.dtype.code == kDLInt && d.dtype.bits == 64) type = DataType::Of(TypeId::Int64);
  else if (d.dtype.code == kDLInt && d.dtype.bits == 32) type = DataType::Of(TypeId::Int32);
  else if (d.dtype.code == kDLBool) type = DataType::Of(TypeId::Bool);
  else return Status::NotSupported("DLPack dtype");
  MemorySpace space = d.device.device_type == kDLCUDA ? MemorySpace::Device : d.device.device_type == kDLCUDAManaged ? MemorySpace::Managed : MemorySpace::Host;
  Column c;
  c.type = type;
  c.length = d.shape[0];
  size_t bytes = TypeByteWidth(type) * c.length;
  auto* hold = new DLManagedTensor*(t);
  c.values = std::make_shared<Buffer>(static_cast<uint8_t*>(d.data) + d.byte_offset, bytes, space,
                                      [](void*, size_t, void* ctx) { auto** p = static_cast<DLManagedTensor**>(ctx); if ((*p)->deleter) (*p)->deleter(*p); delete p; }, hold);
  return c;
}

}  // namespace aster::interop
