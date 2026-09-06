#pragma once
#include <memory>
#include <vector>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"
#include "aster/exec/tile.hpp"

namespace aster::exec {

enum class VectorMetric { L2, Cosine, InnerProduct };

struct AnnQuery {
  std::vector<float> vector;
  uint32_t k = 10;
  VectorMetric metric = VectorMetric::L2;
};

struct AnnResult {
  std::vector<int64_t> row_ids;
  std::vector<float> distances;
};

// Vector columns are first class. The brute force index runs anywhere; the CAGRA index uses cuVS
// when compiled in and is the phase 6 path for ANN similarity joins inside the engine.
class VectorIndex {
 public:
  virtual ~VectorIndex() = default;
  virtual Status Build(const Column& vectors, ExecContext& ctx) = 0;
  virtual Result<AnnResult> Search(const AnnQuery& q, ExecContext& ctx) const = 0;
  virtual const char* name() const = 0;
  virtual uint32_t dims() const = 0;
};

class BruteForceIndex final : public VectorIndex {
 public:
  Status Build(const Column& vectors, ExecContext& ctx) override;
  Result<AnnResult> Search(const AnnQuery& q, ExecContext& ctx) const override;
  const char* name() const override { return "brute_force"; }
  uint32_t dims() const override { return dims_; }

 private:
  Column vectors_;
  uint32_t dims_ = 0;
};

#if ASTER_HAVE_CUVS
class CagraIndex final : public VectorIndex {
 public:
  Status Build(const Column& vectors, ExecContext& ctx) override;
  Result<AnnResult> Search(const AnnQuery& q, ExecContext& ctx) const override;
  const char* name() const override { return "cagra"; }
  uint32_t dims() const override { return dims_; }

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  uint32_t dims_ = 0;
};
#endif

std::unique_ptr<VectorIndex> MakeVectorIndex(bool prefer_gpu);
float VectorDistance(const float* a, const float* b, uint32_t dims, VectorMetric m);

}  // namespace aster::exec
