#pragma once
#include <map>
#include <string>
#include <vector>

#include "aster/common/status.hpp"
#include "aster/integration/plan_ir.hpp"

namespace aster::integration {

enum class Support { Gpu, Cpu, Unsupported };
const char* SupportName(Support s);

struct CapabilityEntry {
  std::string key;      // "rel:join", "fn:equal", "agg:sum", "type:string", "join:semi"
  Support support = Support::Unsupported;
  std::string note;
};

// The registry is data, not code. Coverage gaps are visible and testable; a TSV file overrides defaults.
class CapabilityRegistry {
 public:
  CapabilityRegistry();
  static CapabilityRegistry Defaults();
  Status LoadTsv(const std::string& path);
  Status SaveTsv(const std::string& path) const;
  void Set(const std::string& key, Support s, std::string note = "");
  Support Get(const std::string& key) const;
  Support Function(const std::string& name) const { return Get("fn:" + name); }
  Support Aggregate(const std::string& name) const { return Get("agg:" + name); }
  Support Relation(plan::RelKind k) const;
  Support Type(TypeId t) const;
  Support JoinKind(plan::JoinType t) const;
  // Deepest support level for one node, ignoring children. Reason is filled for anything below Gpu.
  Support CheckNode(const plan::Rel& rel, std::string* reason) const;
  Support CheckExpr(const plan::Expr& e, std::string* reason) const;
  std::vector<CapabilityEntry> Entries() const;
  std::vector<CapabilityEntry> Gaps() const;  // everything not Gpu

 private:
  std::map<std::string, CapabilityEntry> table_;
};

}  // namespace aster::integration
