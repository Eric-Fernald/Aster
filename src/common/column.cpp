#include "aster/common/column.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace aster {

namespace {
void FreeHost(void* p, size_t, void*) { std::free(p); }
void FreeVector(void*, size_t, void* ctx) { delete static_cast<std::vector<uint8_t>*>(ctx); }
inline bool BitGet(const uint8_t* bits, int64_t i) { return (bits[i >> 3] >> (i & 7)) & 1; }
inline void BitSet(uint8_t* bits, int64_t i) { bits[i >> 3] |= uint8_t(1u << (i & 7)); }
}  // namespace

Buffer::~Buffer() {
  if (del_ && data_) del_(data_, size_, ctx_);
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    if (del_ && data_) del_(data_, size_, ctx_);
    data_ = o.data_; size_ = o.size_; space_ = o.space_; del_ = o.del_; ctx_ = o.ctx_;
    o.data_ = nullptr; o.size_ = 0; o.del_ = nullptr; o.ctx_ = nullptr;
  }
  return *this;
}

std::shared_ptr<Buffer> Buffer::AllocateHost(size_t bytes) {
  void* p = std::calloc(bytes ? bytes : 1, 1);
  return std::make_shared<Buffer>(p, bytes, MemorySpace::Host, FreeHost, nullptr);
}

std::shared_ptr<Buffer> Buffer::FromVector(std::vector<uint8_t> v) {
  auto* owned = new std::vector<uint8_t>(std::move(v));
  return std::make_shared<Buffer>(owned->data(), owned->size(), MemorySpace::Host, FreeVector, owned);
}

std::shared_ptr<Buffer> Buffer::CopyOf(const void* data, size_t size) {
  auto b = AllocateHost(size);
  if (size) std::memcpy(b->data(), data, size);
  return b;
}

std::shared_ptr<Buffer> Buffer::Wrap(const void* data, size_t size, MemorySpace space) {
  return std::make_shared<Buffer>(const_cast<void*>(data), size, space, nullptr, nullptr);
}

bool Column::IsValid(int64_t i) const {
  if (!validity) return true;
  return BitGet(validity->data(), i);
}

int64_t Column::dictionary_size() const {
  if (!dictionary_offsets) return 0;
  return static_cast<int64_t>(dictionary_offsets->size() / sizeof(int32_t)) - 1;
}

size_t Column::nbytes() const {
  size_t n = 0;
  if (validity) n += validity->size();
  if (offsets) n += offsets->size();
  if (values) n += values->size();
  return n;
}

std::string_view Column::GetString(int64_t i) const {
  if (is_dictionary_encoded()) return DictionaryEntry(Values<int32_t>()[i]);
  const int32_t* off = offsets->as<int32_t>();
  return std::string_view(reinterpret_cast<const char*>(values->data()) + off[i], off[i + 1] - off[i]);
}

std::string_view Column::DictionaryEntry(int32_t code) const {
  const int32_t* off = dictionary_offsets->as<int32_t>();
  return std::string_view(reinterpret_cast<const char*>(dictionary->data()) + off[code], off[code + 1] - off[code]);
}

Column Column::DecodeDictionary() const {
  if (!is_dictionary_encoded()) return *this;
  std::vector<std::string> out(length);
  std::vector<bool> valid(length);
  for (int64_t i = 0; i < length; ++i) {
    valid[i] = IsValid(i);
    if (valid[i]) out[i] = std::string(GetString(i));
  }
  return MakeStringColumn(out, null_count ? &valid : nullptr);
}

Column Column::Empty(DataType t) {
  Column c;
  c.type = t;
  c.values = Buffer::AllocateHost(0);
  if (!IsFixedWidth(t.id)) {
    c.offsets = Buffer::AllocateHost(sizeof(int32_t));
  }
  return c;
}

std::shared_ptr<Buffer> MakeValidity(const std::vector<bool>& valid, int64_t* null_count) {
  auto b = Buffer::AllocateHost((valid.size() + 7) / 8);
  int64_t nulls = 0;
  for (size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) BitSet(b->data(), static_cast<int64_t>(i)); else ++nulls;
  }
  if (null_count) *null_count = nulls;
  return b;
}

template <typename T>
Column MakeColumn(TypeId id, const std::vector<T>& v, const std::vector<bool>* valid) {
  Column c;
  c.type = DataType::Of(id);
  c.length = static_cast<int64_t>(v.size());
  c.values = Buffer::CopyOf(v.data(), v.size() * sizeof(T));
  if (valid) c.validity = MakeValidity(*valid, &c.null_count);
  return c;
}

template Column MakeColumn<int8_t>(TypeId, const std::vector<int8_t>&, const std::vector<bool>*);
template Column MakeColumn<int16_t>(TypeId, const std::vector<int16_t>&, const std::vector<bool>*);
template Column MakeColumn<int32_t>(TypeId, const std::vector<int32_t>&, const std::vector<bool>*);
template Column MakeColumn<int64_t>(TypeId, const std::vector<int64_t>&, const std::vector<bool>*);
template Column MakeColumn<uint8_t>(TypeId, const std::vector<uint8_t>&, const std::vector<bool>*);
template Column MakeColumn<uint16_t>(TypeId, const std::vector<uint16_t>&, const std::vector<bool>*);
template Column MakeColumn<uint32_t>(TypeId, const std::vector<uint32_t>&, const std::vector<bool>*);
template Column MakeColumn<uint64_t>(TypeId, const std::vector<uint64_t>&, const std::vector<bool>*);
template Column MakeColumn<float>(TypeId, const std::vector<float>&, const std::vector<bool>*);
template Column MakeColumn<double>(TypeId, const std::vector<double>&, const std::vector<bool>*);

Column MakeStringColumn(const std::vector<std::string>& v, const std::vector<bool>* valid) {
  Column c;
  c.type = DataType::Of(TypeId::String);
  c.length = static_cast<int64_t>(v.size());
  std::vector<int32_t> off(v.size() + 1, 0);
  size_t total = 0;
  for (size_t i = 0; i < v.size(); ++i) { total += v[i].size(); off[i + 1] = static_cast<int32_t>(total); }
  c.offsets = Buffer::CopyOf(off.data(), off.size() * sizeof(int32_t));
  c.values = Buffer::AllocateHost(total);
  size_t pos = 0;
  for (const auto& s : v) { std::memcpy(c.values->data() + pos, s.data(), s.size()); pos += s.size(); }
  if (valid) c.validity = MakeValidity(*valid, &c.null_count);
  return c;
}

Column MakeDictionaryColumn(const std::vector<std::string>& dict, const std::vector<int32_t>& codes,
                            uint64_t dict_id, const std::vector<bool>* valid) {
  Column c = MakeColumn<int32_t>(TypeId::String, codes, valid);
  c.type = DataType::Of(TypeId::String);
  Column d = MakeStringColumn(dict);
  c.dictionary = d.values;
  c.dictionary_offsets = d.offsets;
  c.dictionary_id = dict_id;
  return c;
}

size_t RecordBatch::nbytes() const {
  size_t n = 0;
  for (const auto& c : columns) n += c.nbytes();
  return n;
}

Result<int> RecordBatch::ColumnIndex(const std::string& name) const {
  int i = schema.FieldIndex(name);
  if (i < 0) return Status::NotFound("column " + name);
  return i;
}

namespace {
std::string CellToString(const Column& c, int64_t i) {
  if (!c.IsValid(i)) return "NULL";
  switch (c.type.id) {
    case TypeId::Bool: return c.Values<uint8_t>()[i] ? "true" : "false";
    case TypeId::Int8: return std::to_string(c.Values<int8_t>()[i]);
    case TypeId::Int16: return std::to_string(c.Values<int16_t>()[i]);
    case TypeId::Int32: case TypeId::Date32: return std::to_string(c.Values<int32_t>()[i]);
    case TypeId::Int64: case TypeId::Timestamp: case TypeId::Decimal64: return std::to_string(c.Values<int64_t>()[i]);
    case TypeId::UInt8: return std::to_string(c.Values<uint8_t>()[i]);
    case TypeId::UInt16: return std::to_string(c.Values<uint16_t>()[i]);
    case TypeId::UInt32: return std::to_string(c.Values<uint32_t>()[i]);
    case TypeId::UInt64: return std::to_string(c.Values<uint64_t>()[i]);
    case TypeId::Float32: return std::to_string(c.Values<float>()[i]);
    case TypeId::Float64: return std::to_string(c.Values<double>()[i]);
    case TypeId::String: case TypeId::Binary: return std::string(c.GetString(i));
    case TypeId::FixedVector: return "<vec" + std::to_string(c.type.width) + ">";
    default: return "?";
  }
}
}  // namespace

std::string Column::ToString(int64_t max_rows) const {
  std::ostringstream os;
  os << "[";
  for (int64_t i = 0; i < std::min(length, max_rows); ++i) {
    if (i) os << ", ";
    os << CellToString(*this, i);
  }
  if (length > max_rows) os << ", ...";
  os << "]";
  return os.str();
}

std::string RecordBatch::ToString(int64_t max_rows) const {
  std::ostringstream os;
  os << schema.ToString() << "\n";
  for (int64_t r = 0; r < std::min(num_rows(), max_rows); ++r) {
    for (size_t c = 0; c < columns.size(); ++c) {
      if (c) os << " | ";
      os << CellToString(columns[c], r);
    }
    os << "\n";
  }
  if (num_rows() > max_rows) os << "... (" << num_rows() << " rows)\n";
  return os.str();
}

Column SliceColumn(const Column& c, int64_t offset, int64_t length) {
  std::vector<RowIdx> rows(length);
  for (int64_t i = 0; i < length; ++i) rows[i] = static_cast<RowIdx>(offset + i);
  return TakeColumn(c, rows);
}

RecordBatchPtr SliceBatch(const RecordBatch& b, int64_t offset, int64_t length) {
  auto out = std::make_shared<RecordBatch>();
  out->schema = b.schema;
  for (const auto& c : b.columns) out->columns.push_back(SliceColumn(c, offset, length));
  return out;
}

Column TakeColumn(const Column& c, const std::vector<RowIdx>& rows) {
  Column out;
  out.type = c.type;
  out.length = static_cast<int64_t>(rows.size());
  out.dictionary = c.dictionary;
  out.dictionary_offsets = c.dictionary_offsets;
  out.dictionary_id = c.dictionary_id;
  if (c.validity) {
    std::vector<bool> valid(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) valid[i] = c.IsValid(rows[i]);
    out.validity = MakeValidity(valid, &out.null_count);
  }
  if (IsFixedWidth(c.type.id) || c.is_dictionary_encoded()) {
    size_t w = c.is_dictionary_encoded() ? sizeof(int32_t) : TypeByteWidth(c.type);
    out.values = Buffer::AllocateHost(w * rows.size());
    const uint8_t* src = c.values->data();
    uint8_t* dst = out.values->data();
    for (size_t i = 0; i < rows.size(); ++i) std::memcpy(dst + i * w, src + size_t(rows[i]) * w, w);
  } else {
    const int32_t* off = c.offsets->as<int32_t>();
    std::vector<int32_t> noff(rows.size() + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < rows.size(); ++i) { total += off[rows[i] + 1] - off[rows[i]]; noff[i + 1] = static_cast<int32_t>(total); }
    out.offsets = Buffer::CopyOf(noff.data(), noff.size() * sizeof(int32_t));
    out.values = Buffer::AllocateHost(total);
    for (size_t i = 0; i < rows.size(); ++i)
      std::memcpy(out.values->data() + noff[i], c.values->data() + off[rows[i]], off[rows[i] + 1] - off[rows[i]]);
  }
  return out;
}

Column TakeColumn(const Column& c, const SelectionVector& sel) {
  if (sel.all) return c;
  return TakeColumn(c, sel.rows);
}

RecordBatchPtr TakeBatch(const RecordBatch& b, const std::vector<RowIdx>& rows) {
  auto out = std::make_shared<RecordBatch>();
  out->schema = b.schema;
  for (const auto& c : b.columns) out->columns.push_back(TakeColumn(c, rows));
  return out;
}

RecordBatchPtr TakeBatch(const RecordBatch& b, const SelectionVector& sel) {
  if (sel.all) return std::make_shared<RecordBatch>(b);
  return TakeBatch(b, sel.rows);
}

RecordBatchPtr ConcatBatches(const std::vector<RecordBatchPtr>& batches) {
  auto out = std::make_shared<RecordBatch>();
  if (batches.empty()) return out;
  out->schema = batches[0]->schema;
  size_t ncols = batches[0]->columns.size();
  for (size_t ci = 0; ci < ncols; ++ci) {
    const Column& first = batches[0]->columns[ci];
    Column c;
    c.type = first.type;
    c.dictionary = first.dictionary;
    c.dictionary_offsets = first.dictionary_offsets;
    c.dictionary_id = first.dictionary_id;
    int64_t total = 0;
    bool any_valid = false;
    for (const auto& b : batches) { total += b->columns[ci].length; any_valid |= (b->columns[ci].validity != nullptr); }
    c.length = total;
    if (any_valid) {
      std::vector<bool> valid;
      valid.reserve(total);
      for (const auto& b : batches)
        for (int64_t i = 0; i < b->columns[ci].length; ++i) valid.push_back(b->columns[ci].IsValid(i));
      c.validity = MakeValidity(valid, &c.null_count);
    }
    bool fixed = IsFixedWidth(first.type.id) || first.is_dictionary_encoded();
    if (fixed) {
      size_t w = first.is_dictionary_encoded() ? sizeof(int32_t) : TypeByteWidth(first.type);
      c.values = Buffer::AllocateHost(w * total);
      size_t pos = 0;
      for (const auto& b : batches) {
        const Column& src = b->columns[ci];
        std::memcpy(c.values->data() + pos, src.values->data(), w * src.length);
        pos += w * src.length;
      }
    } else {
      std::vector<int32_t> off;
      off.reserve(total + 1);
      off.push_back(0);
      size_t bytes = 0;
      for (const auto& b : batches) {
        const Column& src = b->columns[ci];
        const int32_t* so = src.offsets->as<int32_t>();
        for (int64_t i = 0; i < src.length; ++i) off.push_back(static_cast<int32_t>(bytes + so[i + 1]));
        bytes += so[src.length];
      }
      c.offsets = Buffer::CopyOf(off.data(), off.size() * sizeof(int32_t));
      c.values = Buffer::AllocateHost(bytes);
      size_t pos = 0;
      for (const auto& b : batches) {
        const Column& src = b->columns[ci];
        size_t n = src.offsets->as<int32_t>()[src.length];
        std::memcpy(c.values->data() + pos, src.values->data(), n);
        pos += n;
      }
    }
    out->columns.push_back(std::move(c));
  }
  return out;
}

}  // namespace aster
