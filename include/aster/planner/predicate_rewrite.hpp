#pragma once
#include <map>
#include <optional>

#include "aster/integration/plan_ir.hpp"
#include "aster/storage/encoding.hpp"

namespace aster::planner {

// Per column encoded domain description for one segment chunk.
struct EncodedDomain {
  storage::Encoding encoding = storage::Encoding::Plain;
  const storage::DictionaryView* dictionary = nullptr;
  const storage::ForView* frame = nullptr;
};

// Rewrites predicates so they evaluate on codes or packed values without decoding.
// `WHERE region = 'EU'` becomes `WHERE region_code = 3` for the chunk whose dictionary maps EU to 3.
class PredicateRewriter {
 public:
  static bool IsRewritable(const plan::Expr& pred);
  // Returns a rewritten predicate, or a literal boolean when the chunk can be decided outright.
  static plan::ExprPtr Rewrite(const plan::Expr& pred, const std::map<int, EncodedDomain>& domains);
  static std::optional<bool> ConstantValue(const plan::Expr& e);
};

}  // namespace aster::planner
