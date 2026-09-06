#include "../internal.hpp"

namespace aster::storage {

using namespace internal;

Status EncodePlain(const Column& col, std::vector<uint8_t>& out) {
  ChunkHeader h = MakeHeader(col, Encoding::Plain);
  Put(out, &h, sizeof h);
  PutValidity(out, col);
  if (IsFixedWidth(col.type.id)) {
    Put(out, col.values->data(), TypeByteWidth(col.type) * col.length);
  } else {
    const int32_t* off = col.offsets->as<int32_t>();
    Put(out, off, (col.length + 1) * sizeof(int32_t));
    Put(out, col.values->data(), off[col.length]);
  }
  FinishHeader(out);
  return Status::OK();
}

Result<Column> DecodePlain(const ChunkHeader& h, const uint8_t* payload) {
  Column c;
  c.type = TypeFromHeader(h);
  c.length = h.num_rows;
  const uint8_t* p = payload;
  TakeValidity(c, h, p);
  if (IsFixedWidth(c.type.id)) {
    size_t n = TypeByteWidth(c.type) * h.num_rows;
    c.values = Buffer::CopyOf(p, n);
  } else {
    size_t on = (h.num_rows + 1) * sizeof(int32_t);
    c.offsets = Buffer::CopyOf(p, on);
    p += on;
    int32_t total = c.offsets->as<int32_t>()[h.num_rows];
    c.values = Buffer::CopyOf(p, total);
  }
  return c;
}

}  // namespace aster::storage
