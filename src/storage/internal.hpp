#pragma once
#include <cstring>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/storage/encoding.hpp"

namespace aster::storage::internal {

inline int64_t ReadI64(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::UInt64: return static_cast<int64_t>(c.Values<uint64_t>()[i]);
    default: return c.Values<int64_t>()[i];
  }
}

inline void WriteI64(Column& c, int64_t i, int64_t v) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: c.MutableValues<uint8_t>()[i] = static_cast<uint8_t>(v); break;
    case TypeId::Int8: c.MutableValues<int8_t>()[i] = static_cast<int8_t>(v); break;
    case TypeId::Int16: c.MutableValues<int16_t>()[i] = static_cast<int16_t>(v); break;
    case TypeId::UInt16: c.MutableValues<uint16_t>()[i] = static_cast<uint16_t>(v); break;
    case TypeId::Int32: case TypeId::Date32: c.MutableValues<int32_t>()[i] = static_cast<int32_t>(v); break;
    case TypeId::UInt32: c.MutableValues<uint32_t>()[i] = static_cast<uint32_t>(v); break;
    case TypeId::UInt64: c.MutableValues<uint64_t>()[i] = static_cast<uint64_t>(v); break;
    default: c.MutableValues<int64_t>()[i] = v; break;
  }
}

inline Column AllocFixed(const DataType& t, uint32_t rows) {
  Column c;
  c.type = t;
  c.length = rows;
  c.values = Buffer::AllocateHost(TypeByteWidth(t) * rows);
  return c;
}

inline void Put(std::vector<uint8_t>& out, const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  out.insert(out.end(), b, b + n);
}
template <typename T>
inline void PutVal(std::vector<uint8_t>& out, T v) { Put(out, &v, sizeof v); }
template <typename T>
inline T GetVal(const uint8_t*& p) { T v; std::memcpy(&v, p, sizeof v); p += sizeof v; return v; }

inline size_t ValidityBytes(uint32_t rows) { return (rows + 7) / 8; }

inline void PutValidity(std::vector<uint8_t>& out, const Column& c) {
  if (!c.null_count) return;
  Put(out, c.validity->data(), ValidityBytes(static_cast<uint32_t>(c.length)));
}

inline void TakeValidity(Column& c, const ChunkHeader& h, const uint8_t*& p) {
  if (!h.null_count) return;
  size_t n = ValidityBytes(h.num_rows);
  c.validity = Buffer::CopyOf(p, n);
  c.null_count = h.null_count;
  p += n;
}

inline ChunkHeader MakeHeader(const Column& c, Encoding e) {
  ChunkHeader h;
  h.encoding = static_cast<uint8_t>(e);
  h.type_id = static_cast<uint8_t>(c.type.id);
  h.type_width = c.type.width;
  h.num_rows = static_cast<uint32_t>(c.length);
  h.null_count = static_cast<uint32_t>(c.null_count);
  return h;
}

inline void FinishHeader(std::vector<uint8_t>& out) {
  auto* h = reinterpret_cast<ChunkHeader*>(out.data());
  h->payload_bytes = static_cast<uint32_t>(out.size() - sizeof(ChunkHeader));
}

inline DataType TypeFromHeader(const ChunkHeader& h) {
  DataType t;
  t.id = static_cast<TypeId>(h.type_id);
  t.width = h.type_width;
  return t;
}

}  // namespace aster::storage::internal
