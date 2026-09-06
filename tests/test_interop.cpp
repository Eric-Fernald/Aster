#include "aster/interop/dlpack_export.hpp"
#include "test_framework.hpp"

using namespace aster;
using namespace aster::interop;

ASTER_TEST(dlpack_export_and_import_zero_copy) {
  Column c = MakeColumn<double>(TypeId::Float64, {1.5, 2.5, 3.5});
  ASTER_ASSIGN_OK(DLManagedTensor* t, DlpackExporter::Export(c));
  ASTER_CHECK_EQ(t->dl_tensor.ndim, 1);
  ASTER_CHECK_EQ(t->dl_tensor.shape[0], 3);
  ASTER_CHECK_EQ(int(t->dl_tensor.dtype.code), int(kDLFloat));
  ASTER_CHECK_EQ(int(t->dl_tensor.dtype.bits), 64);
  ASTER_CHECK(t->dl_tensor.data == c.values->data());
  ASTER_CHECK_EQ(int(t->dl_tensor.device.device_type), int(kDLCPU));
  ASTER_ASSIGN_OK(Column back, DlpackExporter::Import(t));
  ASTER_CHECK_EQ(back.length, 3);
  ASTER_CHECK_NEAR(back.Values<double>()[2], 3.5, 1e-12);
}

ASTER_TEST(dlpack_vector_column_is_2d) {
  std::vector<float> v = {1, 2, 3, 4, 5, 6};
  Column c;
  c.type = DataType::Vector(3);
  c.length = 2;
  c.values = Buffer::CopyOf(v.data(), v.size() * sizeof(float));
  ASTER_ASSIGN_OK(DLManagedTensor* t, DlpackExporter::Export(c));
  ASTER_CHECK_EQ(t->dl_tensor.ndim, 2);
  ASTER_CHECK_EQ(t->dl_tensor.shape[1], 3);
  ASTER_CHECK_EQ(t->dl_tensor.strides[0], 3);
  t->deleter(t);
}

ASTER_TEST(dlpack_rejects_nulls_and_strings) {
  std::vector<bool> valid = {true, false};
  Column withnull = MakeColumn<int64_t>(TypeId::Int64, {1, 2}, &valid);
  ASTER_CHECK(!DlpackExporter::Export(withnull).ok());
  ASTER_CHECK(!DlpackExporter::Export(MakeStringColumn({"a"})).ok());
  RecordBatch b;
  b.schema.fields = {{"x", DataType::Of(TypeId::Float32), false}, {"y", DataType::Of(TypeId::Float32), false}};
  b.columns = {MakeColumn<float>(TypeId::Float32, {1, 2, 3}), MakeColumn<float>(TypeId::Float32, {4, 5, 6})};
  ASTER_ASSIGN_OK(DLManagedTensor* m, DlpackExporter::ExportMatrix(b));
  ASTER_CHECK_EQ(m->dl_tensor.shape[0], 3);
  ASTER_CHECK_EQ(m->dl_tensor.shape[1], 2);
  ASTER_CHECK_NEAR(static_cast<float*>(m->dl_tensor.data)[4], 5.0f, 1e-6);
  m->deleter(m);
}
