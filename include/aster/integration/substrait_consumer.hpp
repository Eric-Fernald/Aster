#pragma once
#include <string>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::integration {

struct TableResolver {
  virtual ~TableResolver() = default;
  virtual Result<Schema> ResolveTable(const std::vector<std::string>& names) = 0;
};

// Substrait is the contract. The consumer turns a serialized substrait.Plan into the internal IR.
// Built without protobuf it reports NotSupported so hosts can use PlanBuilder or the JSON form.
class SubstraitConsumer {
 public:
  static bool Available();
  static Result<plan::RelPtr> FromBinary(const std::string& bytes, TableResolver& resolver);
  static Result<plan::RelPtr> FromJson(const std::string& json, TableResolver& resolver);
  // Function URI anchors map to short names used by the capability registry (e.g. "equal", "sum").
  static std::string NormalizeFunctionName(const std::string& substrait_name);
};

}  // namespace aster::integration
