#pragma once
#include <cstdint>
#include <functional>
#include <string>

#include "aster/common/types.hpp"

namespace aster::memory {

enum class Tier : uint8_t { Hbm = 0, Host = 1, Nvme = 2, Object = 3, None = 255 };
const char* TierName(Tier t);

struct PageKey {
  SegmentId segment_id = 0;
  ColumnId column_id = 0;
  bool operator==(const PageKey& o) const { return segment_id == o.segment_id && column_id == o.column_id; }
  bool operator<(const PageKey& o) const {
    return segment_id != o.segment_id ? segment_id < o.segment_id : column_id < o.column_id;
  }
  std::string ToString() const { return std::to_string(segment_id) + ":" + std::to_string(column_id); }
};

struct PageKeyHash {
  size_t operator()(const PageKey& k) const { return std::hash<uint64_t>()(k.segment_id * 1000003ULL + k.column_id); }
};

// Where the encoded bytes live on durable storage.
struct PageLocator {
  std::string path;
  uint64_t offset = 0;
  uint32_t bytes = 0;
  uint32_t checksum = 0;
};

// The page descriptor from the design plan. A page is one encoded column chunk of one segment.
struct PageDesc {
  uint64_t segment_id = 0;
  uint32_t column_id = 0;
  uint32_t encoded_bytes = 0;
  uint8_t tier = static_cast<uint8_t>(Tier::Nvme);
  uint8_t encoding = 0;
  uint16_t pin_count = 0;
  uint64_t last_access_epoch = 0;
  void* device_ptr = nullptr;
  void* host_ptr = nullptr;
  uint64_t access_count = 0;
  bool scheduled = false;     // planner has scheduled this page for an upcoming pipeline
  bool dimension_hint = false;  // small dimension table page, bias toward keeping resident
  PageLocator locator;

  PageKey key() const { return {segment_id, column_id}; }
  Tier current_tier() const { return static_cast<Tier>(tier); }
  bool resident_on_device() const { return device_ptr != nullptr; }
  bool resident_on_host() const { return host_ptr != nullptr; }
};

}  // namespace aster::memory
