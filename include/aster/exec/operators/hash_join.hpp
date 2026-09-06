#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "aster/exec/backend.hpp"
#include "aster/exec/tile.hpp"
#include "aster/memory/spill_controller.hpp"

namespace aster::exec {

struct HashJoinOptions {
  size_t build_budget_bytes = size_t(8) << 30;
  uint32_t radix_bits = 4;   // 16 partitions when the build side does not fit
};

// Build side attempts full residency; if it does not fit, both sides radix partition and partition
// pairs run one at a time with the non resident partitions spilled. Probe side always streams.
class HashJoinBuild {
 public:
  HashJoinBuild(JoinSpec spec, Schema build_schema, Schema output_schema, HashJoinOptions opts = {});
  Status Add(const RecordBatch& batch);
  Status Finish(ExecContext& ctx);
  bool partitioned() const { return partitioned_; }
  uint32_t num_partitions() const { return partitioned_ ? (1u << opts_.radix_bits) : 1; }
  uint64_t build_rows() const { return build_rows_; }
  bool shared_dictionary_keys() const { return shared_dict_; }
  uint64_t dictionary_id() const { return dict_id_; }
  // Probe one tile. Left/anti semantics track matched rows across probes; call FinishProbe for right side extras.
  Result<RecordBatchPtr> Probe(const Tile& tile, OperatorBackend& backend, ExecContext& ctx);
  Result<RecordBatchPtr> Probe(const RecordBatch& probe, OperatorBackend& backend, ExecContext& ctx);
  const Schema& output_schema() const { return out_schema_; }
  std::vector<uint32_t> PartitionRows(const RecordBatch& batch, const std::vector<int>& keys) const;

 private:
  Result<RecordBatchPtr> ProbePartitioned(const RecordBatch& probe, OperatorBackend& backend, ExecContext& ctx);
  JoinSpec spec_;
  Schema build_schema_, out_schema_;
  HashJoinOptions opts_;
  std::vector<RecordBatchPtr> pending_;
  RecordBatchPtr build_;                       // resident build table
  std::vector<RecordBatchPtr> partitions_;     // when partitioned
  std::vector<memory::SpillHandle> spilled_;
  bool partitioned_ = false;
  bool finished_ = false;
  bool shared_dict_ = false;
  uint64_t dict_id_ = 0;
  uint64_t build_rows_ = 0;
  size_t build_bytes_ = 0;
};

// Reference hash table used by the CPU backend and by the fused kernel codegen as the model layout.
struct HostHashTable {
  std::vector<uint64_t> hashes;
  std::vector<int32_t> slots;   // row index or -1
  std::vector<int32_t> next;    // chain
  uint32_t mask = 0;
  void Build(const std::vector<uint64_t>& row_hashes);
  template <typename F>
  void ForEachCandidate(uint64_t h, F&& fn) const {
    int32_t r = slots[h & mask];
    while (r >= 0) { if (hashes[r] == h) fn(r); r = next[r]; }
  }
};

Result<std::vector<uint64_t>> HashRows(const RecordBatch& b, const std::vector<int>& keys);
Result<bool> KeysEqual(const RecordBatch& a, int64_t ra, const std::vector<int>& ka, const RecordBatch& b, int64_t rb, const std::vector<int>& kb);

}  // namespace aster::exec
