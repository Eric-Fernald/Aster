#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aster/exec/tile.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::exec {

enum class AggKind { Sum, Count, CountStar, Min, Max, Avg, CountDistinct };
AggKind ParseAggKind(const std::string& fn);

struct KeyValue {
  bool is_null = false;
  uint8_t kind = 0;  // 0 int, 1 double, 2 string
  int64_t i = 0;
  double d = 0;
  std::string s;
  bool operator==(const KeyValue& o) const {
    if (is_null != o.is_null || kind != o.kind) return false;
    if (is_null) return true;
    return kind == 0 ? i == o.i : kind == 1 ? d == o.d : s == o.s;
  }
};

struct AggState {
  double sum = 0;
  int64_t isum = 0;
  int64_t count = 0;
  double dmin = 0, dmax = 0;
  int64_t imin = 0, imax = 0;
  std::string smin, smax;
  bool seen = false;
  bool is_float = false;
  bool is_string = false;
  std::unique_ptr<std::unordered_set<uint64_t>> distinct;
};

// Streaming hash aggregate: per worker partial states merged at the pipeline breaker.
// Run length inputs aggregate over runs and multiply by the run length.
class HashAggregateState {
 public:
  HashAggregateState(std::vector<plan::ExprPtr> keys, std::vector<plan::AggregateFn> aggs, Schema in_schema, Schema out_schema);
  Status Update(const Tile& tile);
  Status Update(const RecordBatch& batch, const SelectionVector& sel, const std::vector<uint32_t>* run_lengths);
  Status Merge(HashAggregateState& other);
  Result<RecordBatchPtr> Finalize();
  size_t num_groups() const { return groups_.size(); }
  const Schema& output_schema() const { return out_schema_; }
  static Result<RecordBatchPtr> Combine(std::vector<std::unique_ptr<HashAggregateState>>& partials);

 private:
  struct Group {
    std::vector<KeyValue> keys;
    std::vector<AggState> states;
  };
  static KeyValue KeyAt(const Column& c, int64_t row);
  static uint64_t HashKeys(const std::vector<KeyValue>& keys);
  size_t FindOrInsert(std::vector<KeyValue> keys, uint64_t h);
  void Accumulate(AggState& st, AggKind kind, const Column* col, int64_t row, int64_t weight);
  void MergeState(AggState& into, AggState& from, AggKind kind);

  std::vector<plan::ExprPtr> keys_;
  std::vector<plan::AggregateFn> aggs_;
  std::vector<AggKind> kinds_;
  Schema in_schema_, out_schema_;
  std::unordered_multimap<uint64_t, size_t> index_;
  std::vector<Group> groups_;
};

}  // namespace aster::exec
