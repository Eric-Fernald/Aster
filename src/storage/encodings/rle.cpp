#include "../internal.hpp"

namespace aster::storage {

using namespace internal;

Status EncodeRunLength(const Column& col, std::vector<uint8_t>& out) {
  if (!IsIntegerLike(col.type.id) && col.type.id != TypeId::Bool) return Status::NotSupported("RLE needs integer-like column");
  uint8_t w = static_cast<uint8_t>(TypeByteWidth(col.type));
  std::vector<int64_t> vals;
  std::vector<uint32_t> lens;
  for (int64_t i = 0; i < col.length; ++i) {
    int64_t v = col.IsValid(i) ? ReadI64(col, i) : 0;
    if (!vals.empty() && vals.back() == v) ++lens.back();
    else { vals.push_back(v); lens.push_back(1); }
  }
  ChunkHeader h = MakeHeader(col, Encoding::RunLength);
  Put(out, &h, sizeof h);
  PutVal<uint32_t>(out, static_cast<uint32_t>(vals.size()));
  PutVal<uint8_t>(out, w);
  out.insert(out.end(), 3, 0);
  for (int64_t v : vals) Put(out, &v, w);
  Put(out, lens.data(), lens.size() * sizeof(uint32_t));
  PutValidity(out, col);
  FinishHeader(out);
  return Status::OK();
}

Result<Column> DecodeRunLength(const ChunkHeader& h, const uint8_t* payload) {
  const uint8_t* p = payload;
  uint32_t runs = GetVal<uint32_t>(p);
  uint8_t w = GetVal<uint8_t>(p);
  p += 3;
  const uint8_t* vals = p;
  p += size_t(w) * runs;
  const uint32_t* lens = reinterpret_cast<const uint32_t*>(p);
  p += sizeof(uint32_t) * runs;
  Column c = AllocFixed(TypeFromHeader(h), h.num_rows);
  TakeValidity(c, h, p);
  uint32_t row = 0;
  for (uint32_t r = 0; r < runs; ++r) {
    int64_t v = 0;
    switch (w) {
      case 1: { int8_t x; std::memcpy(&x, vals + r, 1); v = x; break; }
      case 2: { int16_t x; std::memcpy(&x, vals + r * 2, 2); v = x; break; }
      case 4: { int32_t x; std::memcpy(&x, vals + r * 4, 4); v = x; break; }
      default: std::memcpy(&v, vals + r * 8, 8); break;
    }
    for (uint32_t k = 0; k < lens[r] && row < h.num_rows; ++k) WriteI64(c, row++, v);
  }
  return c;
}

}  // namespace aster::storage
