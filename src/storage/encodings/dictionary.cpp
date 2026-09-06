#include <unordered_map>

#include "../internal.hpp"

namespace aster::storage {

using namespace internal;

Status EncodeDictionary(const Column& col, const EncodeOptions& opts, std::vector<uint8_t>& out) {
  const bool is_str = !IsFixedWidth(col.type.id);
  std::vector<std::string> dict;
  std::unordered_map<std::string, int32_t> index;
  if (opts.shared_dictionary) {
    dict = *opts.shared_dictionary;
    for (size_t i = 0; i < dict.size(); ++i) index[dict[i]] = static_cast<int32_t>(i);
  }
  std::vector<int32_t> codes(col.length, 0);
  size_t w = is_str ? 0 : TypeByteWidth(col.type);
  for (int64_t i = 0; i < col.length; ++i) {
    if (!col.IsValid(i)) continue;
    std::string key = is_str ? std::string(col.GetString(i))
                             : std::string(reinterpret_cast<const char*>(col.values->data() + i * w), w);
    auto it = index.find(key);
    if (it == index.end()) {
      if (opts.shared_dictionary) return Status::Invalid("value missing from shared dictionary");
      it = index.emplace(key, static_cast<int32_t>(dict.size())).first;
      dict.push_back(key);
    }
    codes[i] = it->second;
  }
  uint8_t code_width = dict.size() <= 256 ? 1 : dict.size() <= 65536 ? 2 : 4;

  ChunkHeader h = MakeHeader(col, Encoding::Dictionary);
  Put(out, &h, sizeof h);
  PutVal<uint32_t>(out, static_cast<uint32_t>(dict.size()));
  PutVal<uint8_t>(out, code_width);
  out.insert(out.end(), 3, 0);
  std::vector<int32_t> off(dict.size() + 1, 0);
  for (size_t i = 0; i < dict.size(); ++i) off[i + 1] = off[i] + static_cast<int32_t>(dict[i].size());
  Put(out, off.data(), off.size() * sizeof(int32_t));
  for (const auto& s : dict) Put(out, s.data(), s.size());
  PutValidity(out, col);
  for (int32_t c : codes) {
    if (code_width == 1) PutVal<uint8_t>(out, static_cast<uint8_t>(c));
    else if (code_width == 2) PutVal<uint16_t>(out, static_cast<uint16_t>(c));
    else PutVal<int32_t>(out, c);
  }
  FinishHeader(out);
  return Status::OK();
}

Result<Column> DecodeDictionary(const ChunkHeader& h, const uint8_t* payload, uint64_t dictionary_id) {
  const uint8_t* p = payload;
  uint32_t count = GetVal<uint32_t>(p);
  uint8_t code_width = GetVal<uint8_t>(p);
  p += 3;
  const int32_t* off = reinterpret_cast<const int32_t*>(p);
  p += (count + 1) * sizeof(int32_t);
  const uint8_t* dict_bytes = p;
  p += off[count];
  DataType t = TypeFromHeader(h);
  Column c;
  c.type = t;
  c.length = h.num_rows;
  TakeValidity(c, h, p);
  std::vector<int32_t> codes(h.num_rows);
  for (uint32_t i = 0; i < h.num_rows; ++i) {
    if (code_width == 1) codes[i] = p[i];
    else if (code_width == 2) { uint16_t v; std::memcpy(&v, p + i * 2, 2); codes[i] = v; }
    else std::memcpy(&codes[i], p + i * 4, 4);
  }
  if (!IsFixedWidth(t.id)) {
    // Strings stay in the encoded domain: codes plus dictionary.
    c.values = Buffer::CopyOf(codes.data(), codes.size() * sizeof(int32_t));
    c.dictionary = Buffer::CopyOf(dict_bytes, off[count]);
    c.dictionary_offsets = Buffer::CopyOf(off, (count + 1) * sizeof(int32_t));
    c.dictionary_id = dictionary_id;
    return c;
  }
  size_t w = TypeByteWidth(t);
  c.values = Buffer::AllocateHost(w * h.num_rows);
  for (uint32_t i = 0; i < h.num_rows; ++i)
    if (c.IsValid(i)) std::memcpy(c.values->data() + i * w, dict_bytes + off[codes[i]], w);
  return c;
}

}  // namespace aster::storage
