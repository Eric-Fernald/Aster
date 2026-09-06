#include "aster/exec/operators/hash_join.hpp"

#include <algorithm>

#include "aster/common/hash.hpp"
#include "aster/common/log.hpp"
#include "aster/metrics/metrics.hpp"

namespace aster::exec {

namespace {
inline bool IsStr(TypeId t) { return t == TypeId::String || t == TypeId::Binary; }
int64_t IntAt(const Column& c, int64_t i) {
  switch (c.type.id) {
    case TypeId::Bool: case TypeId::UInt8: return c.Values<uint8_t>()[i];
    case TypeId::Int8: return c.Values<int8_t>()[i];
    case TypeId::Int16: return c.Values<int16_t>()[i];
    case TypeId::UInt16: return c.Values<uint16_t>()[i];
    case TypeId::Int32: case TypeId::Date32: return c.Values<int32_t>()[i];
    case TypeId::UInt32: return c.Values<uint32_t>()[i];
    case TypeId::Float32: return static_cast<int64_t>(c.Values<float>()[i]);
    case TypeId::Float64: { double d = c.Values<double>()[i]; int64_t v; std::memcpy(&v, &d, 8); return v; }
    default: return c.Values<int64_t>()[i];
  }
}
}  // namespace

void HostHashTable::Build(const std::vector<uint64_t>& row_hashes) {
  hashes = row_hashes;
  size_t cap = 16;
  while (cap < row_hashes.size() * 2) cap <<= 1;
  mask = static_cast<uint32_t>(cap - 1);
  slots.assign(cap, -1);
  next.assign(row_hashes.size(), -1);
  for (size_t r = 0; r < row_hashes.size(); ++r) {
    uint32_t s = static_cast<uint32_t>(row_hashes[r] & mask);
    next[r] = slots[s];
    slots[s] = static_cast<int32_t>(r);
  }
}

Result<std::vector<uint64_t>> HashRows(const RecordBatch& b, const std::vector<int>& keys) {
  std::vector<uint64_t> h(b.num_rows(), 0x9e3779b97f4a7c15ULL);
  for (int k : keys) {
    if (k < 0 || k >= static_cast<int>(b.columns.size())) return Status::Invalid("join key out of range");
    const Column& c = b.columns[k];
    for (int64_t i = 0; i < b.num_rows(); ++i) {
      uint64_t v;
      if (!c.IsValid(i)) v = 0xdeadbeef;
      else if (c.is_dictionary_encoded() || IsStr(c.type.id)) v = Hash64(c.GetString(i));
      else v = HashInt(static_cast<uint64_t>(IntAt(c, i)));
      h[i] = HashCombine(h[i], v);
    }
  }
  return h;
}

Result<bool> KeysEqual(const RecordBatch& a, int64_t ra, const std::vector<int>& ka, const RecordBatch& b, int64_t rb, const std::vector<int>& kb) {
  for (size_t i = 0; i < ka.size(); ++i) {
    const Column& ca = a.columns[ka[i]];
    const Column& cb = b.columns[kb[i]];
    if (!ca.IsValid(ra) || !cb.IsValid(rb)) return false;
    if (ca.is_dictionary_encoded() && cb.is_dictionary_encoded() && ca.dictionary_id && ca.dictionary_id == cb.dictionary_id) {
      if (ca.Values<int32_t>()[ra] != cb.Values<int32_t>()[rb]) return false;
    } else if (IsStr(ca.type.id) || IsStr(cb.type.id)) {
      if (ca.GetString(ra) != cb.GetString(rb)) return false;
    } else if (ca.type.id == TypeId::Float64 || cb.type.id == TypeId::Float64 || ca.type.id == TypeId::Float32 || cb.type.id == TypeId::Float32) {
      double x = ca.type.id == TypeId::Float64 ? ca.Values<double>()[ra] : ca.type.id == TypeId::Float32 ? ca.Values<float>()[ra] : double(IntAt(ca, ra));
      double y = cb.type.id == TypeId::Float64 ? cb.Values<double>()[rb] : cb.type.id == TypeId::Float32 ? cb.Values<float>()[rb] : double(IntAt(cb, rb));
      if (x != y) return false;
    } else if (IntAt(ca, ra) != IntAt(cb, rb)) {
      return false;
    }
  }
  return true;
}

HashJoinBuild::HashJoinBuild(JoinSpec spec, Schema build_schema, Schema output_schema, HashJoinOptions opts)
    : spec_(std::move(spec)), build_schema_(std::move(build_schema)), out_schema_(std::move(output_schema)), opts_(opts) {}

Status HashJoinBuild::Add(const RecordBatch& batch) {
  if (finished_) return Status::Invalid("join build already finished");
  pending_.push_back(std::make_shared<RecordBatch>(batch));
  build_rows_ += batch.num_rows();
  build_bytes_ += batch.nbytes();
  return Status::OK();
}

std::vector<uint32_t> HashJoinBuild::PartitionRows(const RecordBatch& batch, const std::vector<int>& keys) const {
  auto h = HashRows(batch, keys);
  std::vector<uint32_t> out(batch.num_rows(), 0);
  if (!h.ok()) return out;
  uint32_t mask = (1u << opts_.radix_bits) - 1;
  for (int64_t i = 0; i < batch.num_rows(); ++i) out[i] = static_cast<uint32_t>((h.value()[i] >> 40) & mask);
  return out;
}

Status HashJoinBuild::Finish(ExecContext& ctx) {
  if (finished_) return Status::OK();
  finished_ = true;
  build_ = ConcatBatches(pending_);
  pending_.clear();
  if (!build_->columns.empty() && !spec_.right_keys.empty()) {
    const Column& k = build_->columns[spec_.right_keys[0]];
    shared_dict_ = k.is_dictionary_encoded() && k.dictionary_id != 0;
    dict_id_ = k.dictionary_id;
  }
  size_t budget = ctx.memory ? std::min(opts_.build_budget_bytes, ctx.memory->hbm_free()) : opts_.build_budget_bytes;
  if (build_bytes_ > budget && build_->num_rows() > 0) {
    // Radix partition the build side and spill everything but partition 0.
    partitioned_ = true;
    uint32_t np = 1u << opts_.radix_bits;
    std::vector<std::vector<RowIdx>> rows(np);
    std::vector<uint32_t> part = PartitionRows(*build_, spec_.right_keys);
    for (int64_t i = 0; i < build_->num_rows(); ++i) rows[part[i]].push_back(static_cast<RowIdx>(i));
    partitions_.resize(np);
    for (uint32_t p = 0; p < np; ++p) partitions_[p] = TakeBatch(*build_, rows[p]);
    build_.reset();
    if (ctx.memory) {
      for (uint32_t p = 1; p < np; ++p) {
        // Spill the concatenated value buffers; partitions reload one at a time during probe.
        size_t bytes = partitions_[p]->nbytes();
        std::vector<uint8_t> blob;
        blob.reserve(bytes);
        for (const auto& c : partitions_[p]->columns) {
          if (c.validity) blob.insert(blob.end(), c.validity->data(), c.validity->data() + c.validity->size());
          if (c.offsets) blob.insert(blob.end(), c.offsets->data(), c.offsets->data() + c.offsets->size());
          if (c.values) blob.insert(blob.end(), c.values->data(), c.values->data() + c.values->size());
        }
        ASTER_ASSIGN_OR_RETURN(auto h, ctx.memory->spill().SpillHost(blob.data(), blob.size(), memory::SpillKind::HashTable, p));
        spilled_.push_back(h);
        metrics::Registry::Global().counter("spill_bytes").Add(blob.size());
      }
    }
    ASTER_LOG(Info, "hash join build %llu rows partitioned into %u parts", (unsigned long long)build_rows_, np);
  }
  return Status::OK();
}

Result<RecordBatchPtr> HashJoinBuild::Probe(const Tile& tile, OperatorBackend& backend, ExecContext& ctx) {
  RecordBatchPtr probe = tile.Materialize();
  return Probe(*probe, backend, ctx);
}

namespace {
Result<RecordBatchPtr> ProbeOne(const RecordBatch& build, const RecordBatch& probe, const JoinSpec& spec, const Schema& out_schema) {
  ASTER_ASSIGN_OR_RETURN(auto bh, HashRows(build, spec.right_keys));
  ASTER_ASSIGN_OR_RETURN(auto ph, HashRows(probe, spec.left_keys));
  HostHashTable table;
  table.Build(bh);
  std::vector<RowIdx> lrows, rrows;
  std::vector<bool> matched_probe(probe.num_rows(), false);
  for (int64_t i = 0; i < probe.num_rows(); ++i) {
    bool any = false;
    table.ForEachCandidate(ph[i], [&](int32_t r) {
      auto eq = KeysEqual(probe, i, spec.left_keys, build, r, spec.right_keys);
      if (eq.ok() && eq.value()) {
        any = true;
        if (spec.type == plan::JoinType::Inner || spec.type == plan::JoinType::Left) { lrows.push_back(static_cast<RowIdx>(i)); rrows.push_back(static_cast<RowIdx>(r)); }
      }
    });
    matched_probe[i] = any;
    if (spec.type == plan::JoinType::Semi && any) lrows.push_back(static_cast<RowIdx>(i));
    if (spec.type == plan::JoinType::Anti && !any) lrows.push_back(static_cast<RowIdx>(i));
    if (spec.type == plan::JoinType::Left && !any) { lrows.push_back(static_cast<RowIdx>(i)); rrows.push_back(UINT32_MAX); }
  }
  auto out = std::make_shared<RecordBatch>();
  out->schema = out_schema;
  RecordBatchPtr left = TakeBatch(probe, lrows);
  for (auto& c : left->columns) out->columns.push_back(std::move(c));
  if (spec.type == plan::JoinType::Inner || spec.type == plan::JoinType::Left) {
    // Right side gather with null rows for unmatched left rows.
    std::vector<RowIdx> safe(rrows.size());
    std::vector<bool> valid(rrows.size());
    for (size_t i = 0; i < rrows.size(); ++i) { valid[i] = rrows[i] != UINT32_MAX; safe[i] = valid[i] ? rrows[i] : 0; }
    RecordBatchPtr right = build.num_rows() ? TakeBatch(build, safe) : nullptr;
    for (size_t ci = 0; ci < build.columns.size(); ++ci) {
      Column c = right ? right->columns[ci] : Column::Empty(build.columns[ci].type);
      if (!right) { c.length = static_cast<int64_t>(rrows.size()); c.values = Buffer::AllocateHost(std::max<size_t>(1, TypeByteWidth(c.type)) * rrows.size()); if (c.offsets) c.offsets = Buffer::AllocateHost((rrows.size() + 1) * 4); }
      if (spec.type == plan::JoinType::Left) {
        std::vector<bool> v(rrows.size());
        for (size_t i = 0; i < rrows.size(); ++i) v[i] = valid[i] && c.IsValid(static_cast<int64_t>(i));
        c.validity = MakeValidity(v, &c.null_count);
      }
      out->columns.push_back(std::move(c));
    }
  }
  return out;
}
}  // namespace

Result<RecordBatchPtr> HashJoinBuild::ProbePartitioned(const RecordBatch& probe, OperatorBackend& backend, ExecContext& ctx) {
  uint32_t np = num_partitions();
  std::vector<uint32_t> part = PartitionRows(probe, spec_.left_keys);
  std::vector<std::vector<RowIdx>> rows(np);
  for (int64_t i = 0; i < probe.num_rows(); ++i) rows[part[i]].push_back(static_cast<RowIdx>(i));
  std::vector<RecordBatchPtr> outs;
  for (uint32_t p = 0; p < np; ++p) {
    if (rows[p].empty() && spec_.type != plan::JoinType::Anti) continue;
    RecordBatchPtr sub = TakeBatch(probe, rows[p]);
    ASTER_ASSIGN_OR_RETURN(auto r, ProbeOne(*partitions_[p], *sub, spec_, out_schema_));
    outs.push_back(r);
  }
  return backend.Concat(outs, ctx);
}

Result<RecordBatchPtr> HashJoinBuild::Probe(const RecordBatch& probe, OperatorBackend& backend, ExecContext& ctx) {
  if (!finished_) ASTER_RETURN_NOT_OK(Finish(ctx));
  if (partitioned_) return ProbePartitioned(probe, backend, ctx);
  return ProbeOne(*build_, probe, spec_, out_schema_);
}

}  // namespace aster::exec
