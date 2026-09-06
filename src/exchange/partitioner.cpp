#include "aster/exchange/partitioner.hpp"

#include "aster/exec/operators/hash_join.hpp"

namespace aster::exchange {

Result<std::vector<uint32_t>> HashPartitioner::Assign(const RecordBatch& b) const {
  ASTER_ASSIGN_OR_RETURN(auto h, exec::HashRows(b, keys_));
  std::vector<uint32_t> out(b.num_rows());
  for (int64_t i = 0; i < b.num_rows(); ++i) out[i] = static_cast<uint32_t>(h[i] % n_);
  return out;
}

Result<std::vector<RecordBatchPtr>> HashPartitioner::Split(const RecordBatch& b) const {
  ASTER_ASSIGN_OR_RETURN(auto part, Assign(b));
  std::vector<std::vector<RowIdx>> rows(n_);
  for (int64_t i = 0; i < b.num_rows(); ++i) rows[part[i]].push_back(static_cast<RowIdx>(i));
  std::vector<RecordBatchPtr> out(n_);
  for (uint32_t p = 0; p < n_; ++p) out[p] = TakeBatch(b, rows[p]);
  return out;
}

}  // namespace aster::exchange
