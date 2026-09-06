#pragma once
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"

namespace aster::exchange {

// Hash partitions a batch on key columns into N parts. Copartitioned tables share the same
// partitioner (same keys, same N) so their joins never need an exchange.
class HashPartitioner {
 public:
  HashPartitioner(std::vector<int> keys, uint32_t num_partitions) : keys_(std::move(keys)), n_(num_partitions) {}
  Result<std::vector<uint32_t>> Assign(const RecordBatch& b) const;
  Result<std::vector<RecordBatchPtr>> Split(const RecordBatch& b) const;
  uint32_t num_partitions() const { return n_; }
  const std::vector<int>& keys() const { return keys_; }

 private:
  std::vector<int> keys_;
  uint32_t n_;
};

}  // namespace aster::exchange
