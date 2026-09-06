#include "../internal.hpp"

#if ASTER_HAVE_LZ4
#include <lz4.h>
#endif
#if ASTER_HAVE_ZSTD
#include <zstd.h>
#endif

namespace aster::storage {

using namespace internal;

bool CodecAvailable(Codec c) {
  switch (c) {
    case Codec::None: return true;
#if ASTER_HAVE_LZ4
    case Codec::Lz4: return true;
#endif
#if ASTER_HAVE_ZSTD
    case Codec::Zstd: return true;
#endif
    default: return false;
  }
}

Status CompressGeneral(const std::vector<uint8_t>& inner, Codec codec, std::vector<uint8_t>& out) {
  if (!CodecAvailable(codec)) return Status::NotSupported("codec not compiled in");
  ChunkHeader ih;
  std::memcpy(&ih, inner.data(), sizeof ih);
  ChunkHeader h = ih;
  h.encoding = static_cast<uint8_t>(Encoding::General);
  h.inner = ih.encoding;
  h.codec = static_cast<uint8_t>(codec);
  h.raw_bytes = static_cast<uint32_t>(inner.size());
  Put(out, &h, sizeof h);
  size_t pos = out.size();
  switch (codec) {
#if ASTER_HAVE_LZ4
    case Codec::Lz4: {
      int cap = LZ4_compressBound(static_cast<int>(inner.size()));
      out.resize(pos + cap);
      int n = LZ4_compress_default(reinterpret_cast<const char*>(inner.data()), reinterpret_cast<char*>(out.data() + pos),
                                   static_cast<int>(inner.size()), cap);
      if (n <= 0) return Status::Internal("lz4 compress failed");
      out.resize(pos + n);
      break;
    }
#endif
#if ASTER_HAVE_ZSTD
    case Codec::Zstd: {
      size_t cap = ZSTD_compressBound(inner.size());
      out.resize(pos + cap);
      size_t n = ZSTD_compress(out.data() + pos, cap, inner.data(), inner.size(), 3);
      if (ZSTD_isError(n)) return Status::Internal("zstd compress failed");
      out.resize(pos + n);
      break;
    }
#endif
    default:
      Put(out, inner.data(), inner.size());
      break;
  }
  FinishHeader(out);
  return Status::OK();
}

Result<std::vector<uint8_t>> DecompressGeneral(const ChunkHeader& h, const uint8_t* payload) {
  std::vector<uint8_t> raw(h.raw_bytes);
  switch (static_cast<Codec>(h.codec)) {
#if ASTER_HAVE_LZ4
    case Codec::Lz4: {
      int n = LZ4_decompress_safe(reinterpret_cast<const char*>(payload), reinterpret_cast<char*>(raw.data()),
                                  static_cast<int>(h.payload_bytes), static_cast<int>(h.raw_bytes));
      if (n != static_cast<int>(h.raw_bytes)) return Status::Corrupt("lz4 decompress failed");
      break;
    }
#endif
#if ASTER_HAVE_ZSTD
    case Codec::Zstd: {
      size_t n = ZSTD_decompress(raw.data(), h.raw_bytes, payload, h.payload_bytes);
      if (ZSTD_isError(n) || n != h.raw_bytes) return Status::Corrupt("zstd decompress failed");
      break;
    }
#endif
    case Codec::None:
      std::memcpy(raw.data(), payload, h.raw_bytes);
      break;
    default:
      return Status::NotSupported("codec not compiled in");
  }
  return raw;
}

}  // namespace aster::storage
