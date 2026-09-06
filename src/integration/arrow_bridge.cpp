#include "aster/integration/arrow_bridge.hpp"

namespace aster::integration {

bool ArrowAvailable() {
#if ASTER_HAVE_ARROW
  return true;
#else
  return false;
#endif
}

#if ASTER_HAVE_ARROW
namespace {
struct ArrowHold { std::shared_ptr<arrow::Buffer> buf; };
void ReleaseArrow(void*, size_t, void* ctx) { delete static_cast<ArrowHold*>(ctx); }
std::shared_ptr<Buffer> WrapArrow(const std::shared_ptr<arrow::Buffer>& b, int64_t offset_bytes = 0) {
  if (!b) return nullptr;
  return std::make_shared<Buffer>(const_cast<uint8_t*>(b->data()) + offset_bytes, b->size() - offset_bytes,
                                  MemorySpace::Host, ReleaseArrow, new ArrowHold{b});
}
struct AsterHold { std::shared_ptr<Buffer> buf; };
std::shared_ptr<arrow::Buffer> WrapAster(const std::shared_ptr<Buffer>& b) {
  if (!b) return nullptr;
  auto hold = std::make_shared<AsterHold>(AsterHold{b});
  return std::shared_ptr<arrow::Buffer>(new arrow::Buffer(b->data(), b->size()), [hold](arrow::Buffer* p) { delete p; });
}
}  // namespace

Result<DataType> FromArrow(const arrow::DataType& t) {
  switch (t.id()) {
    case arrow::Type::BOOL: return DataType::Of(TypeId::Bool);
    case arrow::Type::INT8: return DataType::Of(TypeId::Int8);
    case arrow::Type::INT16: return DataType::Of(TypeId::Int16);
    case arrow::Type::INT32: return DataType::Of(TypeId::Int32);
    case arrow::Type::INT64: return DataType::Of(TypeId::Int64);
    case arrow::Type::UINT8: return DataType::Of(TypeId::UInt8);
    case arrow::Type::UINT16: return DataType::Of(TypeId::UInt16);
    case arrow::Type::UINT32: return DataType::Of(TypeId::UInt32);
    case arrow::Type::UINT64: return DataType::Of(TypeId::UInt64);
    case arrow::Type::FLOAT: return DataType::Of(TypeId::Float32);
    case arrow::Type::DOUBLE: return DataType::Of(TypeId::Float64);
    case arrow::Type::DATE32: return DataType::Of(TypeId::Date32);
    case arrow::Type::TIMESTAMP: return DataType::Of(TypeId::Timestamp);
    case arrow::Type::STRING: case arrow::Type::LARGE_STRING: return DataType::Of(TypeId::String);
    case arrow::Type::BINARY: return DataType::Of(TypeId::Binary);
    case arrow::Type::DECIMAL128: {
      const auto& d = static_cast<const arrow::Decimal128Type&>(t);
      if (d.precision() > 18) return Status::NotSupported("decimal precision > 18");
      return DataType::Decimal(d.precision(), d.scale());
    }
    case arrow::Type::FIXED_SIZE_LIST: {
      const auto& l = static_cast<const arrow::FixedSizeListType&>(t);
      if (l.value_type()->id() != arrow::Type::FLOAT) return Status::NotSupported("only float32 vectors");
      return DataType::Vector(l.list_size());
    }
    case arrow::Type::DICTIONARY: return FromArrow(*static_cast<const arrow::DictionaryType&>(t).value_type());
    default: return Status::NotSupported("arrow type " + t.ToString());
  }
}

Result<std::shared_ptr<arrow::DataType>> ToArrow(const DataType& t) {
  switch (t.id) {
    case TypeId::Bool: return arrow::boolean();
    case TypeId::Int8: return arrow::int8();
    case TypeId::Int16: return arrow::int16();
    case TypeId::Int32: return arrow::int32();
    case TypeId::Int64: return arrow::int64();
    case TypeId::UInt8: return arrow::uint8();
    case TypeId::UInt16: return arrow::uint16();
    case TypeId::UInt32: return arrow::uint32();
    case TypeId::UInt64: return arrow::uint64();
    case TypeId::Float32: return arrow::float32();
    case TypeId::Float64: return arrow::float64();
    case TypeId::Date32: return arrow::date32();
    case TypeId::Timestamp: return arrow::timestamp(arrow::TimeUnit::MICRO);
    case TypeId::String: return arrow::utf8();
    case TypeId::Binary: return arrow::binary();
    case TypeId::Decimal64: return arrow::decimal128(t.precision, t.width);
    case TypeId::FixedVector: return arrow::fixed_size_list(arrow::float32(), t.width);
    default: return Status::NotSupported("type to arrow");
  }
}

Result<Column> FromArrow(const std::shared_ptr<arrow::Array>& arr) {
  const auto& data = *arr->data();
  Column c;
  ASTER_ASSIGN_OR_RETURN(c.type, FromArrow(*arr->type()));
  c.length = arr->length();
  c.null_count = arr->null_count();
  if (data.offset != 0) {
    // Slice by copying; zero copy needs aligned offsets.
    auto copied = arrow::Concatenate({arr});
    if (!copied.ok()) return Status::Internal(copied.status().ToString());
    return FromArrow(*copied);
  }
  if (c.null_count > 0) c.validity = WrapArrow(data.buffers[0]);
  if (arr->type()->id() == arrow::Type::DICTIONARY) {
    const auto& dict = static_cast<const arrow::DictionaryArray&>(*arr);
    auto codes = arrow::compute::Cast(*dict.indices(), arrow::int32());
    if (!codes.ok()) return Status::Internal(codes.status().ToString());
    c.values = WrapArrow(codes->make_array()->data()->buffers[1]);
    auto d = dict.dictionary();
    c.dictionary = WrapArrow(d->data()->buffers[2]);
    c.dictionary_offsets = WrapArrow(d->data()->buffers[1]);
    return c;
  }
  if (arr->type()->id() == arrow::Type::BOOL) {
    c.values = Buffer::AllocateHost(c.length);
    const auto& b = static_cast<const arrow::BooleanArray&>(*arr);
    for (int64_t i = 0; i < c.length; ++i) c.values->data()[i] = b.Value(i);
    return c;
  }
  if (arr->type()->id() == arrow::Type::LARGE_STRING) {
    auto casted = arrow::compute::Cast(*arr, arrow::utf8());
    if (!casted.ok()) return Status::Internal(casted.status().ToString());
    return FromArrow(casted->make_array());
  }
  if (arr->type()->id() == arrow::Type::DECIMAL128) {
    c.values = Buffer::AllocateHost(c.length * 8);
    const auto& d = static_cast<const arrow::Decimal128Array&>(*arr);
    for (int64_t i = 0; i < c.length; ++i) c.MutableValues<int64_t>()[i] = static_cast<int64_t>(arrow::Decimal128(d.GetValue(i)).low_bits());
    return c;
  }
  if (arr->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    const auto& l = static_cast<const arrow::FixedSizeListArray&>(*arr);
    c.values = WrapArrow(l.values()->data()->buffers[1]);
    return c;
  }
  if (IsFixedWidth(c.type.id)) {
    c.values = WrapArrow(data.buffers[1]);
  } else {
    c.offsets = WrapArrow(data.buffers[1]);
    c.values = WrapArrow(data.buffers[2]);
  }
  return c;
}

Result<std::shared_ptr<arrow::Array>> ToArrow(const Column& col) {
  ASTER_ASSIGN_OR_RETURN(auto type, ToArrow(col.type));
  std::shared_ptr<arrow::Buffer> validity = col.null_count ? WrapAster(col.validity) : nullptr;
  if (col.is_dictionary_encoded()) {
    auto dict_data = arrow::ArrayData::Make(arrow::utf8(), col.dictionary_size(), {nullptr, WrapAster(col.dictionary_offsets), WrapAster(col.dictionary)});
    auto idx_data = arrow::ArrayData::Make(arrow::int32(), col.length, {validity, WrapAster(col.values)}, col.null_count);
    auto dtype = arrow::dictionary(arrow::int32(), arrow::utf8());
    return std::static_pointer_cast<arrow::Array>(std::make_shared<arrow::DictionaryArray>(dtype, arrow::MakeArray(idx_data), arrow::MakeArray(dict_data)));
  }
  if (col.type.id == TypeId::Bool) {
    arrow::BooleanBuilder b;
    for (int64_t i = 0; i < col.length; ++i) {
      if (!col.IsValid(i)) { auto s = b.AppendNull(); (void)s; }
      else { auto s = b.Append(col.Values<uint8_t>()[i] != 0); (void)s; }
    }
    std::shared_ptr<arrow::Array> out;
    auto s = b.Finish(&out);
    if (!s.ok()) return Status::Internal(s.ToString());
    return out;
  }
  if (col.type.id == TypeId::Decimal64) {
    arrow::Decimal128Builder b(type);
    for (int64_t i = 0; i < col.length; ++i) {
      if (!col.IsValid(i)) { auto s = b.AppendNull(); (void)s; }
      else { auto s = b.Append(arrow::Decimal128(col.Values<int64_t>()[i])); (void)s; }
    }
    std::shared_ptr<arrow::Array> out;
    auto s = b.Finish(&out);
    if (!s.ok()) return Status::Internal(s.ToString());
    return out;
  }
  if (col.type.id == TypeId::FixedVector) {
    auto vals = arrow::ArrayData::Make(arrow::float32(), col.length * col.type.width, {nullptr, WrapAster(col.values)});
    auto data = arrow::ArrayData::Make(type, col.length, {validity}, {vals}, col.null_count);
    return arrow::MakeArray(data);
  }
  if (IsFixedWidth(col.type.id)) {
    auto data = arrow::ArrayData::Make(type, col.length, {validity, WrapAster(col.values)}, col.null_count);
    return arrow::MakeArray(data);
  }
  auto data = arrow::ArrayData::Make(type, col.length, {validity, WrapAster(col.offsets), WrapAster(col.values)}, col.null_count);
  return arrow::MakeArray(data);
}

Result<Schema> FromArrow(const std::shared_ptr<arrow::Schema>& schema) {
  Schema s;
  for (const auto& f : schema->fields()) {
    ASTER_ASSIGN_OR_RETURN(DataType t, FromArrow(*f->type()));
    s.fields.push_back({f->name(), t, f->nullable()});
  }
  return s;
}

Result<std::shared_ptr<arrow::Schema>> ToArrow(const Schema& schema) {
  arrow::FieldVector fields;
  for (const auto& f : schema.fields) {
    ASTER_ASSIGN_OR_RETURN(auto t, ToArrow(f.type));
    fields.push_back(arrow::field(f.name, t, f.nullable));
  }
  return arrow::schema(fields);
}

Result<RecordBatchPtr> FromArrow(const std::shared_ptr<arrow::RecordBatch>& batch) {
  auto out = std::make_shared<RecordBatch>();
  ASTER_ASSIGN_OR_RETURN(out->schema, FromArrow(batch->schema()));
  for (const auto& arr : batch->columns()) {
    ASTER_ASSIGN_OR_RETURN(Column c, FromArrow(arr));
    out->columns.push_back(std::move(c));
  }
  return out;
}

Result<std::shared_ptr<arrow::RecordBatch>> ToArrow(const RecordBatch& batch) {
  ASTER_ASSIGN_OR_RETURN(auto schema, ToArrow(batch.schema));
  arrow::ArrayVector arrays;
  for (const auto& c : batch.columns) {
    ASTER_ASSIGN_OR_RETURN(auto a, ToArrow(c));
    arrays.push_back(a);
  }
  return arrow::RecordBatch::Make(schema, batch.num_rows(), arrays);
}
#endif

}  // namespace aster::integration
