#pragma once
#include <string>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::exec {

// Host reference evaluator for plan expressions. Dictionary columns are compared on codes when the
// predicate is an equality against a literal; everything else decodes lazily per row.
class ExpressionEvaluator {
 public:
  static Result<Column> Evaluate(const plan::Expr& e, const RecordBatch& input);
  // Evaluates a boolean predicate into a selection vector over `sel`.
  static Result<SelectionVector> Filter(const plan::Expr& pred, const RecordBatch& input, const SelectionVector& sel);
  static Result<Column> Cast(const Column& c, DataType to);
  static bool LikeMatch(std::string_view value, std::string_view pattern);
};

}  // namespace aster::exec
