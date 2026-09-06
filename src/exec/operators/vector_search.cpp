#include "aster/exec/operators/vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace aster::exec {

float VectorDistance(const float* a, const float* b, uint32_t dims, VectorMetric m) {
  double acc = 0, na = 0, nb = 0;
  for (uint32_t i = 0; i < dims; ++i) {
    switch (m) {
      case VectorMetric::L2: { double d = double(a[i]) - b[i]; acc += d * d; break; }
      case VectorMetric::InnerProduct: acc += double(a[i]) * b[i]; break;
      case VectorMetric::Cosine: acc += double(a[i]) * b[i]; na += double(a[i]) * a[i]; nb += double(b[i]) * b[i]; break;
    }
  }
  switch (m) {
    case VectorMetric::L2: return static_cast<float>(std::sqrt(acc));
    case VectorMetric::InnerProduct: return static_cast<float>(-acc);
    default: return static_cast<float>(1.0 - acc / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
  }
}

Status BruteForceIndex::Build(const Column& vectors, ExecContext&) {
  if (vectors.type.id != TypeId::FixedVector) return Status::Invalid("vector index needs a FixedVector column");
  vectors_ = vectors;
  dims_ = vectors.type.width;
  return Status::OK();
}

Result<AnnResult> BruteForceIndex::Search(const AnnQuery& q, ExecContext&) const {
  if (q.vector.size() != dims_) return Status::Invalid("query dims mismatch");
  using P = std::pair<float, int64_t>;
  std::priority_queue<P> heap;  // max heap on distance keeps the k smallest
  const float* base = vectors_.Values<float>();
  for (int64_t i = 0; i < vectors_.length; ++i) {
    if (!vectors_.IsValid(i)) continue;
    float d = VectorDistance(q.vector.data(), base + i * dims_, dims_, q.metric);
    if (heap.size() < q.k) heap.push({d, i});
    else if (d < heap.top().first) { heap.pop(); heap.push({d, i}); }
  }
  AnnResult r;
  while (!heap.empty()) { r.row_ids.push_back(heap.top().second); r.distances.push_back(heap.top().first); heap.pop(); }
  std::reverse(r.row_ids.begin(), r.row_ids.end());
  std::reverse(r.distances.begin(), r.distances.end());
  return r;
}

#if ASTER_HAVE_CUVS
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

struct CagraIndex::Impl {
  raft::device_resources res;
  std::unique_ptr<cuvs::neighbors::cagra::index<float, uint32_t>> index;
};

Status CagraIndex::Build(const Column& vectors, ExecContext& ctx) {
  if (vectors.type.id != TypeId::FixedVector) return Status::Invalid("vector index needs a FixedVector column");
  dims_ = vectors.type.width;
  impl_ = std::make_shared<Impl>();
  try {
    auto dataset = raft::make_device_matrix<float, int64_t>(impl_->res, vectors.length, dims_);
    raft::copy(dataset.data_handle(), vectors.Values<float>(), size_t(vectors.length) * dims_, impl_->res.get_stream());
    cuvs::neighbors::cagra::index_params params;
    impl_->index = std::make_unique<cuvs::neighbors::cagra::index<float, uint32_t>>(
        cuvs::neighbors::cagra::build(impl_->res, params, raft::make_const_mdspan(dataset.view())));
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cagra build: ") + e.what());
  }
  return Status::OK();
}

Result<AnnResult> CagraIndex::Search(const AnnQuery& q, ExecContext&) const {
  if (!impl_ || !impl_->index) return Status::Invalid("index not built");
  try {
    auto queries = raft::make_device_matrix<float, int64_t>(impl_->res, 1, dims_);
    raft::copy(queries.data_handle(), q.vector.data(), dims_, impl_->res.get_stream());
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(impl_->res, 1, q.k);
    auto distances = raft::make_device_matrix<float, int64_t>(impl_->res, 1, q.k);
    cuvs::neighbors::cagra::search_params sp;
    cuvs::neighbors::cagra::search(impl_->res, sp, *impl_->index, raft::make_const_mdspan(queries.view()), neighbors.view(), distances.view());
    std::vector<uint32_t> n(q.k); std::vector<float> d(q.k);
    raft::copy(n.data(), neighbors.data_handle(), q.k, impl_->res.get_stream());
    raft::copy(d.data(), distances.data_handle(), q.k, impl_->res.get_stream());
    impl_->res.sync_stream();
    AnnResult r;
    for (uint32_t i = 0; i < q.k; ++i) { r.row_ids.push_back(n[i]); r.distances.push_back(d[i]); }
    return r;
  } catch (const std::exception& e) {
    return Status::Internal(std::string("cagra search: ") + e.what());
  }
}
#endif

std::unique_ptr<VectorIndex> MakeVectorIndex(bool prefer_gpu) {
#if ASTER_HAVE_CUVS
  if (prefer_gpu) return std::make_unique<CagraIndex>();
#else
  (void)prefer_gpu;
#endif
  return std::make_unique<BruteForceIndex>();
}

}  // namespace aster::exec
