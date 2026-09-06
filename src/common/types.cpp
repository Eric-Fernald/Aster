#include "aster/common/types.hpp"

namespace aster {

size_t TypeByteWidth(const DataType& t) {
  switch (t.id) {
    case TypeId::Bool: case TypeId::Int8: case TypeId::UInt8: return 1;
    case TypeId::Int16: case TypeId::UInt16: return 2;
    case TypeId::Int32: case TypeId::UInt32: case TypeId::Float32: case TypeId::Date32: return 4;
    case TypeId::Int64: case TypeId::UInt64: case TypeId::Float64: case TypeId::Decimal64: case TypeId::Timestamp: return 8;
    case TypeId::FixedVector: return 4 * t.width;
    default: return 0;
  }
}

const char* TypeName(TypeId t) {
  static const char* names[] = {"null", "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                                "f32", "f64", "decimal64", "date32", "timestamp", "string", "binary", "vector"};
  return names[static_cast<int>(t)];
}

std::string TypeToString(const DataType& t) {
  std::string s = TypeName(t.id);
  if (t.id == TypeId::FixedVector) s += "[" + std::to_string(t.width) + "]";
  if (t.id == TypeId::Decimal64) s += "(" + std::to_string(t.precision) + "," + std::to_string(t.width) + ")";
  return s;
}

int Schema::FieldIndex(const std::string& name) const {
  for (size_t i = 0; i < fields.size(); ++i)
    if (fields[i].name == name) return static_cast<int>(i);
  return -1;
}

std::string Schema::ToString() const {
  std::string s;
  for (const auto& f : fields) {
    if (!s.empty()) s += ", ";
    s += f.name + ":" + TypeToString(f.type);
  }
  return s;
}

}  // namespace aster
