#include "aster/storage/encoding.hpp"

#include <cstring>

#include "internal.hpp"

namespace aster::storage {

using namespace internal;

const char* EncodingName(Encoding e) {
  switch (e) {
    case Encoding::Plain: return "plain";
    case Encoding::Dictionary: return "dictionary";
    case Encoding::ForBitPack: return "for_bitpack";
    case Encoding::RunLength: return "rle";
    case Encoding::Delta: return "delta";
    case Encoding::General: return "general";
  }
  return "?";
}

uint8_t BitsNeeded(uint64_t max_value) {
  uint8_t b = 0;
  while (max_value) { ++b; max_value >>= 1; }
  return b;
}

size_t PackedBytes(uint32_t n, uint8_t bit_width) { return (size_t(n) * bit_width + 7) / 8; }

void PackBits(const uint64_t* values, uint32_t n, uint8_t bw, uint8_t* out) {
  std::memset(out, 0, PackedBytes(n, bw));
  if (bw == 0) return;
  for (uint32_t i = 0; i < n; ++i) {
    uint64_t v = values[i] & (bw == 64 ? ~0ULL : ((1ULL << bw) - 1));
    size_t bit = size_t(i) * bw;
    for (uint8_t k = 0; k < bw; ++k, ++bit)
      if ((v >> k) & 1) out[bit >> 3] |= uint8_t(1u << (bit & 7));
  }
}

uint64_t UnpackBit(const uint8_t* packed, uint32_t idx, uint8_t bw) {
  if (bw == 0) return 0;
  uint64_t v = 0;
  size_t bit = size_t(idx) * bw;
  for (uint8_t k = 0; k < bw; ++k, ++bit)
    if ((packed[bit >> 3] >> (bit & 7)) & 1) v |= (1ULL << k);
  return v;
}

Result<ChunkHeader> ParseHeader(const uint8_t* data, size_t len) {
  if (len < sizeof(ChunkHeader)) return Status::Corrupt("chunk too small");
  ChunkHeader h;
  std::memcpy(&h, data, sizeof h);
  if (h.magic != 0xA5E7) return Status::Corrupt("bad chunk magic");
  if (sizeof(ChunkHeader) + h.payload_bytes > len) return Status::Corrupt("chunk payload truncated");
  return h;
}

Result<EncodedChunk> EncodeColumn(const Column& col, const EncodeOptions& opts) {
  EncodedChunk out;
  out.type = col.type;
  out.num_rows = static_cast<uint32_t>(col.length);
  out.null_count = static_cast<uint32_t>(col.null_count);
  out.dictionary_id = opts.dictionary_id;
  out.zone = ZoneMap::Compute(col);
  Column src = col.is_dictionary_encoded() && opts.encoding != Encoding::Dictionary ? col.DecodeDictionary() : col;
  std::vector<uint8_t> inner;
  Status s;
  switch (opts.encoding) {
    case Encoding::Plain: s = EncodePlain(src, inner); break;
    case Encoding::Dictionary: s = EncodeDictionary(src, opts, inner); break;
    case Encoding::ForBitPack: s = EncodeForBitPack(src, inner); break;
    case Encoding::RunLength: s = EncodeRunLength(src, inner); break;
    case Encoding::Delta: s = EncodeDelta(src, inner); break;
    case Encoding::General: s = EncodePlain(src, inner); break;
  }
  if (!s.ok()) return s;
  out.encoding = opts.encoding == Encoding::General ? Encoding::Plain : opts.encoding;
  out.inner = out.encoding;
  if (opts.outer_codec != Codec::None) {
    ASTER_RETURN_NOT_OK(CompressGeneral(inner, opts.outer_codec, out.data));
    out.inner = out.encoding;
    out.encoding = Encoding::General;
    out.codec = opts.outer_codec;
  } else {
    out.data = std::move(inner);
  }
  return out;
}

Result<Column> DecodeChunk(const uint8_t* data, size_t len) {
  ASTER_ASSIGN_OR_RETURN(ChunkHeader h, ParseHeader(data, len));
  const uint8_t* payload = data + sizeof(ChunkHeader);
  switch (static_cast<Encoding>(h.encoding)) {
    case Encoding::Plain: return DecodePlain(h, payload);
    case Encoding::Dictionary: return DecodeDictionary(h, payload, 0);
    case Encoding::ForBitPack: return DecodeForBitPack(h, payload);
    case Encoding::RunLength: return DecodeRunLength(h, payload);
    case Encoding::Delta: return DecodeDelta(h, payload);
    case Encoding::General: {
      ASTER_ASSIGN_OR_RETURN(auto raw, DecompressGeneral(h, payload));
      return DecodeChunk(raw.data(), raw.size());
    }
  }
  return Status::Corrupt("unknown encoding");
}

Result<Column> DecodeChunk(const EncodedChunk& chunk) {
  ASTER_ASSIGN_OR_RETURN(Column c, DecodeChunk(chunk.data.data(), chunk.data.size()));
  if (c.is_dictionary_encoded()) c.dictionary_id = chunk.dictionary_id;
  return c;
}

int32_t DictionaryView::CodeAt(uint32_t row) const {
  switch (code_width) {
    case 1: return codes[row];
    case 2: { uint16_t v; std::memcpy(&v, codes + row * 2, 2); return v; }
    default: { int32_t v; std::memcpy(&v, codes + row * 4, 4); return v; }
  }
}

int32_t DictionaryView::LookupCode(std::string_view v) const {
  for (size_t i = 0; i < values.size(); ++i)
    if (values[i] == v) return static_cast<int32_t>(i);
  return -1;
}

uint64_t ForView::PackedAt(uint32_t row) const { return UnpackBit(packed, row, bit_width); }

int64_t RunView::ValueAt(uint32_t run) const {
  int64_t v = 0;
  switch (width) {
    case 1: { int8_t x; std::memcpy(&x, run_values + run, 1); v = x; break; }
    case 2: { int16_t x; std::memcpy(&x, run_values + run * 2, 2); v = x; break; }
    case 4: { int32_t x; std::memcpy(&x, run_values + run * 4, 4); v = x; break; }
    default: std::memcpy(&v, run_values + run * 8, 8); break;
  }
  return v;
}

Result<DictionaryView> ViewDictionary(const uint8_t* data, size_t len) {
  ASTER_ASSIGN_OR_RETURN(ChunkHeader h, ParseHeader(data, len));
  if (h.encoding != static_cast<uint8_t>(Encoding::Dictionary)) return Status::Invalid("not dictionary encoded");
  const uint8_t* p = data + sizeof(ChunkHeader);
  DictionaryView v;
  uint32_t count = GetVal<uint32_t>(p);
  v.code_width = GetVal<uint8_t>(p);
  p += 3;
  const int32_t* off = reinterpret_cast<const int32_t*>(p);
  p += (count + 1) * sizeof(int32_t);
  v.values.reserve(count);
  for (uint32_t i = 0; i < count; ++i) v.values.emplace_back(reinterpret_cast<const char*>(p) + off[i], off[i + 1] - off[i]);
  p += off[count];
  if (h.null_count) { v.validity = p; p += ValidityBytes(h.num_rows); }
  v.codes = p;
  v.num_rows = h.num_rows;
  return v;
}

Result<ForView> ViewFor(const uint8_t* data, size_t len) {
  ASTER_ASSIGN_OR_RETURN(ChunkHeader h, ParseHeader(data, len));
  if (h.encoding != static_cast<uint8_t>(Encoding::ForBitPack)) return Status::Invalid("not FOR encoded");
  const uint8_t* p = data + sizeof(ChunkHeader);
  ForView v;
  v.reference = GetVal<int64_t>(p);
  v.bit_width = GetVal<uint8_t>(p);
  p += 7;
  if (h.null_count) { v.validity = p; p += ValidityBytes(h.num_rows); }
  v.packed = p;
  v.num_rows = h.num_rows;
  return v;
}

Result<RunView> ViewRuns(const uint8_t* data, size_t len) {
  ASTER_ASSIGN_OR_RETURN(ChunkHeader h, ParseHeader(data, len));
  if (h.encoding != static_cast<uint8_t>(Encoding::RunLength)) return Status::Invalid("not RLE encoded");
  const uint8_t* p = data + sizeof(ChunkHeader);
  RunView v;
  v.num_runs = GetVal<uint32_t>(p);
  v.width = GetVal<uint8_t>(p);
  p += 3;
  v.run_values = p;
  p += size_t(v.width) * v.num_runs;
  v.run_lengths = reinterpret_cast<const uint32_t*>(p);
  p += sizeof(uint32_t) * v.num_runs;
  if (h.null_count) v.validity = p;
  v.num_rows = h.num_rows;
  return v;
}

}  // namespace aster::storage
