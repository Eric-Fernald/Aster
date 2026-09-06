#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/types.hpp"

namespace aster::plan {

enum class ExprKind { Literal, ColumnRef, Call, Cast };

struct NullLiteral {};
using LiteralValue = std::variant<NullLiteral, bool, int64_t, double, std::string>;

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Expr {
  ExprKind kind = ExprKind::Literal;
  DataType type;
  LiteralValue literal;         // Literal
  int field_index = -1;         // ColumnRef into the input relation's output schema
  std::string function;         // Call: Substrait function name (equal, lt, add, and, like, ...)
  std::vector<ExprPtr> args;    // Call or Cast operand
  // Set by the planner when rewritten into the encoded domain.
  bool encoded_domain = false;
  uint64_t dictionary_id = 0;

  std::string ToString() const;
  bool is_literal() const { return kind == ExprKind::Literal; }
  bool is_column() const { return kind == ExprKind::ColumnRef; }
  bool is_call(const char* fn) const { return kind == ExprKind::Call && function == fn; }
};

ExprPtr Lit(int64_t v);
ExprPtr Lit(double v);
ExprPtr Lit(std::string v);
ExprPtr Lit(bool v);
ExprPtr LitNull(DataType t);
ExprPtr Col(int index, DataType t = {});
ExprPtr Call(std::string fn, std::vector<ExprPtr> args, DataType out = {});
ExprPtr Cast(ExprPtr e, DataType to);

enum class RelKind { Read, Filter, Project, Join, Aggregate, Sort, Limit, Window, Exchange, ExternalInput };
enum class JoinType { Inner, Left, Right, Full, Semi, Anti };
enum class Placement { Unassigned, Gpu, Cpu };
enum class ExchangeKind { HashPartition, Broadcast, Gather };

struct AggregateFn {
  std::string function;  // sum, count, min, max, avg, count_distinct
  std::vector<ExprPtr> args;
  bool distinct = false;
  DataType out_type;
};

struct SortKey {
  ExprPtr expr;
  bool ascending = true;
  bool nulls_first = false;
};

struct WindowFn {
  std::string function;  // row_number, rank, sum, ...
  std::vector<ExprPtr> args;
  std::vector<ExprPtr> partition_by;
  std::vector<SortKey> order_by;
  DataType out_type;
};

struct Rel;
using RelPtr = std::shared_ptr<Rel>;

struct Rel {
  RelKind kind = RelKind::Read;
  int node_id = 0;
  Schema output;
  std::vector<RelPtr> inputs;
  Placement placement = Placement::Unassigned;
  std::string fallback_reason;

  // Read
  std::string table;
  std::vector<int> projection;
  ExprPtr pushed_filter;
  // Filter
  ExprPtr predicate;
  // Project
  std::vector<ExprPtr> exprs;
  // Join
  JoinType join_type = JoinType::Inner;
  ExprPtr condition;
  std::vector<int> left_keys, right_keys;
  // Aggregate
  std::vector<ExprPtr> group_keys;
  std::vector<AggregateFn> aggregates;
  // Sort
  std::vector<SortKey> sort_keys;
  // Limit
  int64_t offset = 0, count = -1;
  // Window
  std::vector<WindowFn> windows;
  // Exchange
  ExchangeKind exchange_kind = ExchangeKind::HashPartition;
  std::vector<int> exchange_keys;
  // ExternalInput: batches produced outside the GPU path (CPU fallback or host system).
  std::vector<RecordBatchPtr> external_batches;

  std::string ToString(int indent = 0) const;
  size_t num_inputs() const { return inputs.size(); }
  bool is_pipeline_breaker() const {
    return kind == RelKind::Aggregate || kind == RelKind::Sort || kind == RelKind::Join || kind == RelKind::Window ||
           kind == RelKind::Exchange;
  }
};

const char* RelKindName(RelKind k);
const char* JoinTypeName(JoinType t);
void AssignNodeIds(const RelPtr& root);
void Walk(const RelPtr& root, const std::function<void(const RelPtr&)>& fn);
std::vector<RelPtr> Collect(const RelPtr& root, RelKind kind);

// Programmatic plan construction, used by tests, tools and hosts without a Substrait producer.
class PlanBuilder {
 public:
  static RelPtr Read(std::string table, Schema schema, std::vector<int> projection = {});
  static RelPtr External(Schema schema, std::vector<RecordBatchPtr> batches);
  static RelPtr Filter(RelPtr in, ExprPtr predicate);
  static RelPtr Project(RelPtr in, std::vector<ExprPtr> exprs, std::vector<std::string> names = {});
  static RelPtr Join(RelPtr l, RelPtr r, JoinType t, std::vector<int> lkeys, std::vector<int> rkeys);
  static RelPtr Aggregate(RelPtr in, std::vector<ExprPtr> keys, std::vector<AggregateFn> aggs,
                          std::vector<std::string> names = {});
  static RelPtr Sort(RelPtr in, std::vector<SortKey> keys);
  static RelPtr Limit(RelPtr in, int64_t offset, int64_t count);
  static RelPtr Window(RelPtr in, std::vector<WindowFn> fns, std::vector<std::string> names = {});
  static RelPtr Exchange(RelPtr in, ExchangeKind kind, std::vector<int> keys);
};

DataType InferType(const Expr& e, const Schema& input);
DataType AggregateOutputType(const std::string& fn, DataType in);

}  // namespace aster::plan
