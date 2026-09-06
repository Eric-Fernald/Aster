#include "../internal.hpp"

namespace aster::storage {

using namespace internal;

Status EncodeDelta(const Column& col, std::vector<uint8_t>& out) {
  if (!IsIntegerLike(col.type.id)) return Status::NotSupported("delta needs integer-like column");
  std::vector<int64_t> v(col.length, 0);
  int64_t last = 0;
  for (int64_t i = 0; i < col.length; ++i) { v[i] = col.IsValid(i) ? ReadI64(col, i) : last; last = v[i]; }
  int64_t first = col.length ? v[0] : 0;
  int64_t min_d = 0, max_d = 0;
  for (int64_t i = 1; i < col.length; ++i) {
    int64_t d = v[i] - v[i - 1];
    if (i == 1) { min_d = max_d = d; } else { min_d = std::min(min_d, d); max_d = std::max(max_d, d); }
  }
  uint8_t bw = col.length > 1 ? BitsNeeded(static_cast<uint64_t>(max_d - min_d)) : 0;
  std::vector<uint64_t> packed(col.length > 1 ? col.length - 1 : 0);
  for (int64_t i = 1; i < col.length; ++i) packed[i - 1] = static_cast<uint64_t>((v[i] - v[i - 1]) - min_d);

  ChunkHeader h = MakeHeader(col, Encoding::Delta);
  Put(out, &h, sizeof h);
  PutVal<int64_t>(out, first);
  PutVal<int64_t>(out, min_d);
  PutVal<uint8_t>(out, bw);
  out.insert(out.end(), 7, 0);
  PutValidity(out, col);
  size_t pos = out.size();
  out.resize(pos + PackedBytes(static_cast<uint32_t>(packed.size()), bw));
  PackBits(packed.data(), static_cast<uint32_t>(packed.size()), bw, out.data() + pos);
  FinishHeader(out);
  return Status::OK();
}

Result<Column> DecodeDelta(const ChunkHeader& h, const uint8_t* payload) {
  const uint8_t* p = payload;
  int64_t first = GetVal<int64_t>(p);
  int64_t min_d = GetVal<int64_t>(p);
  uint8_t bw = GetVal<uint8_t>(p);
  p += 7;
  Column c = AllocFixed(TypeFromHeader(h), h.num_rows);
  TakeValidity(c, h, p);
  int64_t cur = first;
  if (h.num_rows) WriteI64(c, 0, cur);
  for (uint32_t i = 1; i < h.num_rows; ++i) {
    cur += min_d + static_cast<int64_t>(UnpackBit(p, i - 1, bw));
    WriteI64(c, i, cur);
  }
  return c;
}

}  // namespace aster::storage
