#include "aster/planner/pipeline_splitter.hpp"

#include <sstream>

#include "aster/common/hash.hpp"

namespace aster::planner {

using namespace plan;

namespace {

struct Builder {
  std::vector<Pipeline> out;

  // Returns the id of the pipeline whose sink is `rel`.
  int Build(const RelPtr& rel) {
    std::vector<int> deps;
    std::vector<RelPtr> chain;
    RelPtr cur = rel;
    // Walk down through streaming operators until a source or breaker input.
    for (;;) {
      chain.insert(chain.begin(), cur);
      if (cur->inputs.empty()) break;
      // A join's build side (right) is its own pipeline; the probe side streams through the join.
      if (cur->kind == RelKind::Join) deps.push_back(Build(cur->inputs[1]));
      RelPtr in = cur->inputs[0];
      if (in->kind != RelKind::Join && in->is_pipeline_breaker()) { deps.push_back(Build(in)); chain.insert(chain.begin(), in); break; }
      cur = in;
    }
    Pipeline p;
    p.id = static_cast<int>(out.size());
    p.ops = chain;
    p.depends_on = deps;
    p.cpu = false;
    for (const auto& op : chain) p.cpu |= op->placement == Placement::Cpu;
    p.shape_hash = PipelineSplitter::ShapeHash(p);
    p.fusable = PipelineSplitter::IsFusable(p);
    out.push_back(p);
    return p.id;
  }
};

}  // namespace

std::vector<Pipeline> PipelineSplitter::Split(const RelPtr& root) {
  Builder b;
  b.Build(root);
  return b.out;
}

std::string PipelineSplitter::ShapeHash(const Pipeline& p) {
  std::ostringstream os;
  for (const auto& op : p.ops) {
    os << RelKindName(op->kind) << "(";
    for (const auto& f : op->output.fields) os << TypeToString(f.type) << ",";
    os << ")";
    switch (op->kind) {
      case RelKind::Filter: os << op->predicate->ToString(); break;
      case RelKind::Project: for (const auto& e : op->exprs) os << e->ToString() << ";"; break;
      case RelKind::Aggregate:
        for (const auto& k : op->group_keys) os << k->ToString() << ";";
        for (const auto& a : op->aggregates) os << a.function << ":" << TypeToString(a.out_type) << ";";
        break;
      case RelKind::Join: os << JoinTypeName(op->join_type) << op->left_keys.size(); break;
      default: break;
    }
    os << "|";
  }
  std::string s = os.str();
  char buf[32];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(Hash64(s)));
  return buf;
}

bool PipelineSplitter::IsFusable(const Pipeline& p) {
  if (p.cpu || p.ops.empty()) return false;
  const RelPtr& src = p.ops.front();
  if (src->kind != RelKind::Read && src->kind != RelKind::Join) return false;
  bool has_work = false;
  for (size_t i = 1; i < p.ops.size(); ++i) {
    switch (p.ops[i]->kind) {
      case RelKind::Filter: case RelKind::Project: has_work = true; break;
      case RelKind::Aggregate: has_work = true; break;   // partial aggregate fused, finalize is the breaker
      case RelKind::Join: has_work = true; break;        // probe fused
      case RelKind::Limit: break;
      default: return false;                             // sort, window, exchange stay operator at a time
    }
  }
  return has_work;
}

}  // namespace aster::planner
