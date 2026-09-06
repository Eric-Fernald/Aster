#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/storage/segment.hpp"

namespace aster::storage {

// Immutable snapshot of a table's segment list. In flight queries hold a shared_ptr so a compaction
// swap never pulls segments out from under them.
struct SegmentList {
  std::string table;
  uint64_t version = 0;
  std::vector<std::shared_ptr<SegmentMeta>> segments;
  uint64_t total_rows() const;
  uint64_t total_bytes() const;
};
using SegmentListPtr = std::shared_ptr<const SegmentList>;

struct TableInfo {
  std::string name;
  Schema schema;
  std::vector<std::string> partition_keys;
  bool dimension_hint = false;
};

// Local file manifest per table. One node only; multi node needs a real catalog (plan section 8).
class Catalog {
 public:
  explicit Catalog(std::string root_dir);
  Status Open();
  Status CreateTable(const TableInfo& info);
  Status DropTable(const std::string& table);
  bool HasTable(const std::string& table) const;
  Result<TableInfo> GetTable(const std::string& table) const;
  std::vector<std::string> Tables() const;
  Result<SegmentListPtr> Snapshot(const std::string& table) const;
  // Atomically replaces the segment list. Old lists survive until their last holder drops.
  Status SwapSegments(const std::string& table, std::vector<std::shared_ptr<SegmentMeta>> segments);
  Status AppendSegment(const std::string& table, std::shared_ptr<SegmentMeta> seg);
  Status ReplaceSegments(const std::string& table, const std::vector<SegmentId>& remove,
                         std::vector<std::shared_ptr<SegmentMeta>> add);
  SegmentId NextSegmentId() { return next_segment_id_++; }
  std::string SegmentPath(const std::string& table, SegmentId id) const;
  const std::string& root() const { return root_; }

 private:
  Status PersistManifest(const std::string& table, const SegmentList& list) const;
  Status LoadManifest(const std::string& table);
  std::string ManifestPath(const std::string& table) const;

  std::string root_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, TableInfo> tables_;
  std::unordered_map<std::string, SegmentListPtr> lists_;
  std::atomic<SegmentId> next_segment_id_{1};
};

}  // namespace aster::storage
