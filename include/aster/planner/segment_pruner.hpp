#pragma once
#include <memory>
#include <vector>

#include "aster/integration/plan_ir.hpp"
#include "aster/storage/catalog.hpp"

namespace aster::planner {

struct PruneResult {
  std::vector<std::shared_ptr<storage::SegmentMeta>> kept;
  uint64_t pruned_segments = 0;
  uint64_t pruned_rows = 0;
  uint64_t kept_rows = 0;
};

// Consults zone maps and bloom filters before any data moves. Unknown predicates keep the segment.
class SegmentPruner {
 public:
  static bool MayMatch(const plan::Expr& pred, const storage::SegmentMeta& seg, const Schema& read_schema,
                       const std::vector<int>& projection);
  static PruneResult Prune(const std::vector<std::shared_ptr<storage::SegmentMeta>>& segments, const plan::Expr* pred,
                           const Schema& read_schema, const std::vector<int>& projection);
  // Collects conjuncts from a Read's pushed filter and any Filter nodes directly above it.
  static plan::ExprPtr CollectPredicates(const plan::RelPtr& read, const plan::RelPtr& root);
};

}  // namespace aster::planner
