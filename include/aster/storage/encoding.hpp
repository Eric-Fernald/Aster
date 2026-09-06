#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/storage/zone_map.hpp"

namespace aster::storage {

enum class Encoding : uint8_t { Plain = 0, Dictionary = 1, ForBitPack = 2, RunLength = 3, Delta = 4, General = 5 };
enum class Codec : uint8_t { None = 0, Lz4 = 1, Zstd = 2 };
const char* EncodingName(Encoding e);

// Every encoded blob starts with this header so one kernel launch can decode it without CPU parsing.
struct ChunkHeader {
  uint16_t magic = 0xA5E7;
  uint8_t encoding = 0;
  uint8_t inner = 0;      // inner encoding when General wraps another
  uint8_t type_id = 0;
  uint8_t codec = 0;
  uint16_t reserved = 0;
  uint32_t type_width = 0;
  uint32_t num_rows = 0;
  uint32_t null_count = 0;
  uint32_t payload_bytes = 0;
  uint32_t raw_bytes = 0;  // decompressed size for General
};
static_assert(sizeof(ChunkHeader) == 28);

struct EncodedChunk {
  Encoding encoding = Encoding::Plain;
  Encoding inner = Encoding::Plain;
  Codec codec = Codec::None;
  DataType type;
  uint32_t num_rows = 0;
  uint32_t null_count = 0;
  uint64_t dictionary_id = 0;
  std::vector<uint8_t> data;  // header + payload
  ZoneMap zone;

  const ChunkHeader& header() const { return *reinterpret_cast<const ChunkHeader*>(data.data()); }
  size_t bytes() const { return data.size(); }
  double compression_ratio(size_t raw) const { return data.empty() ? 1.0 : double(raw) / double(data.size()); }
};

struct EncodeOptions {
  Encoding encoding = Encoding::Plain;
  Codec outer_codec = Codec::None;
  uint64_t dictionary_id = 0;
  const std::vector<std::string>* shared_dictionary = nullptr;  // join on codes needs a shared dictionary
};

Result<EncodedChunk> EncodeColumn(const Column& col, const EncodeOptions& opts);
Result<Column> DecodeChunk(const uint8_t* data, size_t len);
Result<Column> DecodeChunk(const EncodedChunk& chunk);
Result<ChunkHeader> ParseHeader(const uint8_t* data, size_t len);

// Views that let operators work in the encoded domain without a full decode.
struct DictionaryView {
  std::vector<std::string> values;
  uint8_t code_width = 4;
  const uint8_t* codes = nullptr;
  const uint8_t* validity = nullptr;
  uint32_t num_rows = 0;
  int32_t CodeAt(uint32_t row) const;
  int32_t LookupCode(std::string_view v) const;  // -1 when absent
};
struct ForView {
  int64_t reference = 0;
  uint8_t bit_width = 0;
  const uint8_t* packed = nullptr;
  const uint8_t* validity = nullptr;
  uint32_t num_rows = 0;
  uint64_t PackedAt(uint32_t row) const;
  int64_t ValueAt(uint32_t row) const { return reference + static_cast<int64_t>(PackedAt(row)); }
  bool CanRepresent(int64_t v) const { return v >= reference && (v - reference) < (int64_t(1) << bit_width); }
};
struct RunView {
  uint32_t num_runs = 0;
  uint8_t width = 0;
  const uint8_t* run_values = nullptr;
  const uint32_t* run_lengths = nullptr;
  const uint8_t* validity = nullptr;
  uint32_t num_rows = 0;
  int64_t ValueAt(uint32_t run) const;
};

Result<DictionaryView> ViewDictionary(const uint8_t* data, size_t len);
Result<ForView> ViewFor(const uint8_t* data, size_t len);
Result<RunView> ViewRuns(const uint8_t* data, size_t len);

// Bit packing helpers shared by FOR and Delta.
void PackBits(const uint64_t* values, uint32_t n, uint8_t bit_width, uint8_t* out);
uint64_t UnpackBit(const uint8_t* packed, uint32_t idx, uint8_t bit_width);
uint8_t BitsNeeded(uint64_t max_value);
size_t PackedBytes(uint32_t n, uint8_t bit_width);

// Individual codecs. Each writes header + payload into out.
Status EncodePlain(const Column& col, std::vector<uint8_t>& out);
Result<Column> DecodePlain(const ChunkHeader& h, const uint8_t* payload);
Status EncodeDictionary(const Column& col, const EncodeOptions& opts, std::vector<uint8_t>& out);
Result<Column> DecodeDictionary(const ChunkHeader& h, const uint8_t* payload, uint64_t dictionary_id);
Status EncodeForBitPack(const Column& col, std::vector<uint8_t>& out);
Result<Column> DecodeForBitPack(const ChunkHeader& h, const uint8_t* payload);
Status EncodeRunLength(const Column& col, std::vector<uint8_t>& out);
Result<Column> DecodeRunLength(const ChunkHeader& h, const uint8_t* payload);
Status EncodeDelta(const Column& col, std::vector<uint8_t>& out);
Result<Column> DecodeDelta(const ChunkHeader& h, const uint8_t* payload);
Status CompressGeneral(const std::vector<uint8_t>& inner, Codec codec, std::vector<uint8_t>& out);
Result<std::vector<uint8_t>> DecompressGeneral(const ChunkHeader& h, const uint8_t* payload);
bool CodecAvailable(Codec c);

}  // namespace aster::storage
