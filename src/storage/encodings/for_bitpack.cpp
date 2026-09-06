#include "../internal.hpp"

namespace aster::storage {

using namespace internal;

Status EncodeForBitPack(const Column& col, std::vector<uint8_t>& out) {
  if (!IsIntegerLike(col.type.id) && col.type.id != TypeId::Bool) return Status::NotSupported("FOR needs integer-like column");
  int64_t lo = INT64_MAX, hi = INT64_MIN;
  for (int64_t i = 0; i < col.length; ++i) {
    if (!col.IsValid(i)) continue;
    int64_t v = ReadI64(col, i);
    lo = std::min(lo, v); hi = std::max(hi, v);
  }
  if (lo > hi) { lo = 0; hi = 0; }
  uint8_t bw = BitsNeeded(static_cast<uint64_t>(hi - lo));
  std::vector<uint64_t> packed(col.length, 0);
  for (int64_t i = 0; i < col.length; ++i)
    if (col.IsValid(i)) packed[i] = static_cast<uint64_t>(ReadI64(col, i) - lo);

  ChunkHeader h = MakeHeader(col, Encoding::ForBitPack);
  Put(out, &h, sizeof h);
  PutVal<int64_t>(out, lo);
  PutVal<uint8_t>(out, bw);
  out.insert(out.end(), 7, 0);
  PutValidity(out, col);
  size_t pos = out.size();
  out.resize(pos + PackedBytes(static_cast<uint32_t>(col.length), bw));
  PackBits(packed.data(), static_cast<uint32_t>(col.length), bw, out.data() + pos);
  FinishHeader(out);
  return Status::OK();
}

Result<Column> DecodeForBitPack(const ChunkHeader& h, const uint8_t* payload) {
  const uint8_t* p = payload;
  int64_t ref = GetVal<int64_t>(p);
  uint8_t bw = GetVal<uint8_t>(p);
  p += 7;
  Column c = AllocFixed(TypeFromHeader(h), h.num_rows);
  TakeValidity(c, h, p);
  for (uint32_t i = 0; i < h.num_rows; ++i) WriteI64(c, i, ref + static_cast<int64_t>(UnpackBit(p, i, bw)));
  return c;
}

}  // namespace aster::storage
