#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aster {

enum class TypeId : uint8_t {
  Null, Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
  Float32, Float64, Decimal64, Date32, Timestamp, String, Binary, FixedVector
};

struct DataType {
  TypeId id = TypeId::Null;
  uint32_t width = 0;      // FixedVector element count, or Decimal scale
  uint32_t precision = 0;

  static DataType Of(TypeId t) { return {t, 0, 0}; }
  static DataType Vector(uint32_t dims) { return {TypeId::FixedVector, dims, 0}; }
  static DataType Decimal(uint32_t precision, uint32_t scale) { return {TypeId::Decimal64, scale, precision}; }
  bool operator==(const DataType& o) const { return id == o.id && width == o.width && precision == o.precision; }
  bool operator!=(const DataType& o) const { return !(*this == o); }
};

inline bool IsFixedWidth(TypeId t) { return t != TypeId::String && t != TypeId::Binary && t != TypeId::Null; }
inline bool IsIntegral(TypeId t) { return t >= TypeId::Int8 && t <= TypeId::UInt64; }
inline bool IsNumeric(TypeId t) {
  return IsIntegral(t) || t == TypeId::Float32 || t == TypeId::Float64 || t == TypeId::Decimal64;
}
inline bool IsTemporal(TypeId t) { return t == TypeId::Date32 || t == TypeId::Timestamp; }
inline bool IsIntegerLike(TypeId t) { return IsIntegral(t) || IsTemporal(t) || t == TypeId::Decimal64; }
size_t TypeByteWidth(const DataType& t);
const char* TypeName(TypeId t);
std::string TypeToString(const DataType& t);

struct Field {
  std::string name;
  DataType type;
  bool nullable = true;
};

struct Schema {
  std::vector<Field> fields;
  int FieldIndex(const std::string& name) const;
  size_t size() const { return fields.size(); }
  std::string ToString() const;
};

using SegmentId = uint64_t;
using ColumnId = uint32_t;
using RowIdx = uint32_t;

}  // namespace aster
