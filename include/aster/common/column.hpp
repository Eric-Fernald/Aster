#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/common/types.hpp"

namespace aster {

enum class MemorySpace : uint8_t { Host, HostPinned, Device, Managed };

// Owning byte buffer. Device buffers are released through the deleter the HAL supplies.
class Buffer {
 public:
  using Deleter = void (*)(void*, size_t, void* ctx);
  Buffer() = default;
  Buffer(void* data, size_t size, MemorySpace space, Deleter del = nullptr, void* ctx = nullptr)
      : data_(data), size_(size), space_(space), del_(del), ctx_(ctx) {}
  ~Buffer();
  Buffer(Buffer&& o) noexcept { *this = std::move(o); }
  Buffer& operator=(Buffer&& o) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  static std::shared_ptr<Buffer> AllocateHost(size_t bytes);
  static std::shared_ptr<Buffer> FromVector(std::vector<uint8_t> v);
  static std::shared_ptr<Buffer> CopyOf(const void* data, size_t size);
  static std::shared_ptr<Buffer> Wrap(const void* data, size_t size, MemorySpace space);

  uint8_t* data() { return static_cast<uint8_t*>(data_); }
  const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
  template <typename T> T* as() { return reinterpret_cast<T*>(data_); }
  template <typename T> const T* as() const { return reinterpret_cast<const T*>(data_); }
  size_t size() const { return size_; }
  MemorySpace space() const { return space_; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
  MemorySpace space_ = MemorySpace::Host;
  Deleter del_ = nullptr;
  void* ctx_ = nullptr;
};

// Arrow layout column: validity bitmap, offsets for var width, values.
// When dictionary != nullptr, values hold int32 codes and dictionary holds the string payload.
struct Column {
  DataType type;
  int64_t length = 0;
  int64_t null_count = 0;
  std::shared_ptr<Buffer> validity;
  std::shared_ptr<Buffer> offsets;
  std::shared_ptr<Buffer> values;
  std::shared_ptr<Buffer> dictionary;
  std::shared_ptr<Buffer> dictionary_offsets;
  uint64_t dictionary_id = 0;

  bool IsValid(int64_t i) const;
  bool is_dictionary_encoded() const { return dictionary != nullptr; }
  int64_t dictionary_size() const;
  MemorySpace space() const { return values ? values->space() : MemorySpace::Host; }
  size_t nbytes() const;

  template <typename T> const T* Values() const { return values ? values->as<T>() : nullptr; }
  template <typename T> T* MutableValues() { return values ? values->as<T>() : nullptr; }
  std::string_view GetString(int64_t i) const;
  std::string_view DictionaryEntry(int32_t code) const;
  Column DecodeDictionary() const;
  static Column Empty(DataType t);
  std::string ToString(int64_t max_rows = 10) const;
};

template <typename T>
Column MakeColumn(TypeId id, const std::vector<T>& v, const std::vector<bool>* valid = nullptr);
Column MakeStringColumn(const std::vector<std::string>& v, const std::vector<bool>* valid = nullptr);
Column MakeDictionaryColumn(const std::vector<std::string>& dict, const std::vector<int32_t>& codes,
                            uint64_t dict_id, const std::vector<bool>* valid = nullptr);
std::shared_ptr<Buffer> MakeValidity(const std::vector<bool>& valid, int64_t* null_count);

struct SelectionVector {
  std::vector<RowIdx> rows;
  bool all = true;
  int64_t count(int64_t total) const { return all ? total : static_cast<int64_t>(rows.size()); }
  static SelectionVector All() { return {}; }
  static SelectionVector Of(std::vector<RowIdx> r) { SelectionVector s; s.all = false; s.rows = std::move(r); return s; }
};

struct RecordBatch {
  Schema schema;
  std::vector<Column> columns;
  int64_t num_rows() const { return columns.empty() ? 0 : columns[0].length; }
  size_t nbytes() const;
  const Column& column(size_t i) const { return columns.at(i); }
  Result<int> ColumnIndex(const std::string& name) const;
  std::string ToString(int64_t max_rows = 10) const;
};

using RecordBatchPtr = std::shared_ptr<RecordBatch>;
RecordBatchPtr ConcatBatches(const std::vector<RecordBatchPtr>& batches);
RecordBatchPtr SliceBatch(const RecordBatch& b, int64_t offset, int64_t length);
Column SliceColumn(const Column& c, int64_t offset, int64_t length);
Column TakeColumn(const Column& c, const SelectionVector& sel);
Column TakeColumn(const Column& c, const std::vector<RowIdx>& rows);
RecordBatchPtr TakeBatch(const RecordBatch& b, const SelectionVector& sel);
RecordBatchPtr TakeBatch(const RecordBatch& b, const std::vector<RowIdx>& rows);

}  // namespace aster
