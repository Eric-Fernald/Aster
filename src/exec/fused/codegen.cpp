#include "aster/exec/fused/codegen.hpp"

#include <sstream>

namespace aster::exec::fused {

using namespace plan;

std::string Codegen::CudaType(const DataType& t) {
  switch (t.id) {
    case TypeId::Bool: return "bool";
    case TypeId::Int8: return "int8_t";
    case TypeId::Int16: return "int16_t";
    case TypeId::Int32: case TypeId::Date32: return "int32_t";
    case TypeId::Int64: case TypeId::Timestamp: case TypeId::Decimal64: return "int64_t";
    case TypeId::UInt8: return "uint8_t";
    case TypeId::UInt16: return "uint16_t";
    case TypeId::UInt32: return "uint32_t";
    case TypeId::UInt64: return "uint64_t";
    case TypeId::Float32: return "float";
    case TypeId::Float64: return "double";
    case TypeId::String: return "int32_t";  // dictionary code in the fused path
    default: return "int64_t";
  }
}

std::string Codegen::Preamble() {
  return R"(
#include <cstdint>
struct ChunkHeader { uint16_t magic; uint8_t encoding, inner, type_id, codec; uint16_t reserved; uint32_t type_width, num_rows, null_count, payload_bytes, raw_bytes; };
struct PageIn { const uint8_t* data; uint32_t bytes; };
struct Outputs { void** cols; uint32_t* count; };
__device__ __forceinline__ bool bit_get(const uint8_t* bits, uint32_t i) { return bits ? ((bits[i >> 3] >> (i & 7)) & 1) : true; }
__device__ __forceinline__ uint64_t unpack(const uint8_t* p, uint32_t idx, uint32_t bw) {
  if (bw == 0) return 0;
  uint64_t bit = (uint64_t)idx * bw; uint64_t byte = bit >> 3; uint32_t shift = bit & 7;
  uint64_t word = 0;
  #pragma unroll
  for (int k = 0; k < 9; ++k) word |= (uint64_t)p[byte + k] << (8 * k);
  uint64_t mask = bw == 64 ? ~0ull : ((1ull << bw) - 1);
  return (word >> shift) & mask;
}
__device__ __forceinline__ const uint8_t* payload(const PageIn& pg) { return pg.data + sizeof(ChunkHeader); }
__device__ __forceinline__ const ChunkHeader* header(const PageIn& pg) { return (const ChunkHeader*)pg.data; }
__device__ __forceinline__ uint64_t hash64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
)";
}

// Each input column is read straight out of its encoded page. Dictionary and FOR columns never
// materialize plain values inside the kernel; predicates rewritten by the planner compare codes.
std::string Codegen::DecodeSnippet(const ColumnBinding& b, const std::string& row) {
  std::ostringstream os;
  std::string pg = "pages[" + std::to_string(b.index) + "]";
  std::string ct = CudaType(b.type);
  switch (b.encoding) {
    case storage::Encoding::Plain:
      os << "({ const ChunkHeader* h = header(" << pg << "); const uint8_t* p = payload(" << pg << ") + (h->null_count ? (h->num_rows + 7) / 8 : 0); ((const " << ct << "*)p)[" << row << "]; })";
      break;
    case storage::Encoding::Dictionary:
      os << "({ const uint8_t* p = payload(" << pg << "); uint32_t cnt = *(const uint32_t*)p; uint8_t cw = p[4]; p += 8 + (cnt + 1) * 4; p += ((const int32_t*)(payload(" << pg << ") + 8))[cnt];"
         << " const ChunkHeader* h = header(" << pg << "); if (h->null_count) p += (h->num_rows + 7) / 8;"
         << " (int32_t)(cw == 1 ? p[" << row << "] : cw == 2 ? ((const uint16_t*)p)[" << row << "] : ((const int32_t*)p)[" << row << "]); })";
      break;
    case storage::Encoding::ForBitPack:
      os << "({ const uint8_t* p = payload(" << pg << "); int64_t ref = *(const int64_t*)p; uint8_t bw = p[8]; p += 16;"
         << " const ChunkHeader* h = header(" << pg << "); if (h->null_count) p += (h->num_rows + 7) / 8;"
         << " (" << ct << ")(ref + (int64_t)unpack(p, " << row << ", bw)); })";
      break;
    default:
      os << "({ (" << ct << ")0; })";
      break;
  }
  return os.str();
}

Result<std::string> Codegen::ExprToCuda(const Expr& e, const std::vector<ColumnBinding>& inputs, const std::string& row) {
  switch (e.kind) {
    case ExprKind::Literal:
      if (auto b = std::get_if<bool>(&e.literal)) return std::string(*b ? "true" : "false");
      if (auto i = std::get_if<int64_t>(&e.literal)) return "(int64_t)" + std::to_string(*i) + "LL";
      if (auto d = std::get_if<double>(&e.literal)) { std::ostringstream os; os.precision(17); os << *d; return os.str(); }
      return Status::NotSupported("string literal in fused kernel; planner must rewrite to dictionary code");
    case ExprKind::ColumnRef: {
      for (const auto& b : inputs)
        if (b.index == e.field_index) {
          if (e.encoded_domain && b.encoding == storage::Encoding::ForBitPack)
            return "({ const uint8_t* p = payload(pages[" + std::to_string(b.index) + "]); uint8_t bw = p[8]; p += 16; const ChunkHeader* h = header(pages[" + std::to_string(b.index) + "]); if (h->null_count) p += (h->num_rows + 7) / 8; unpack(p, " + row + ", bw); })";
          return DecodeSnippet(b, row);
        }
      return "in" + std::to_string(e.field_index) + "[" + row + "]";
    }
    case ExprKind::Cast: {
      ASTER_ASSIGN_OR_RETURN(std::string a, ExprToCuda(*e.args[0], inputs, row));
      return "((" + CudaType(e.type) + ")(" + a + "))";
    }
    case ExprKind::Call: break;
  }
  std::vector<std::string> a;
  for (const auto& arg : e.args) { ASTER_ASSIGN_OR_RETURN(std::string s, ExprToCuda(*arg, inputs, row)); a.push_back(s); }
  const std::string& f = e.function;
  auto bin = [&](const char* op) -> Result<std::string> {
    if (a.size() != 2) return Status::Invalid(f + " arity");
    return "(" + a[0] + " " + op + " " + a[1] + ")";
  };
  if (f == "equal") return bin("==");
  if (f == "not_equal") return bin("!=");
  if (f == "lt") return bin("<");
  if (f == "lte") return bin("<=");
  if (f == "gt") return bin(">");
  if (f == "gte") return bin(">=");
  if (f == "add") return bin("+");
  if (f == "subtract") return bin("-");
  if (f == "multiply") return bin("*");
  if (f == "divide") return "((double)" + a[0] + " / (double)" + a[1] + ")";
  if (f == "modulus") return bin("%");
  if (f == "and" || f == "or") {
    std::string s = "(";
    for (size_t i = 0; i < a.size(); ++i) { if (i) s += f == "and" ? " && " : " || "; s += a[i]; }
    return s + ")";
  }
  if (f == "not") return "(!" + a[0] + ")";
  if (f == "negate") return "(-" + a[0] + ")";
  if (f == "abs") return "(" + a[0] + " < 0 ? -" + a[0] + " : " + a[0] + ")";
  if (f == "between") return "(" + a[0] + " >= " + a[1] + " && " + a[0] + " <= " + a[2] + ")";
  if (f == "in") {
    std::string s = "(";
    for (size_t i = 1; i < a.size(); ++i) { if (i > 1) s += " || "; s += a[0] + " == " + a[i]; }
    return s + ")";
  }
  if (f == "if_then") return "(" + a[0] + " ? " + a[1] + " : " + a[a.size() - 1] + ")";
  if (f == "hash") return "hash64((uint64_t)" + a[0] + ")";
  return Status::NotSupported("function " + f + " has no fused kernel");
}

Result<KernelSpec> Codegen::Generate(const planner::Pipeline& p, const std::vector<ColumnBinding>& inputs, uint32_t tile_rows) {
  KernelSpec spec;
  spec.shape_hash = p.shape_hash;
  spec.name = "aster_pipe_" + p.shape_hash;
  spec.inputs = inputs;
  spec.tile_rows = tile_rows;
  std::ostringstream os;
  os << Preamble();
  os << "extern \"C\" __global__ void " << spec.name << "(const PageIn* pages, uint32_t num_rows, Outputs out, uint64_t* partial_keys, double* partial_vals, uint32_t* partial_counts, uint32_t table_slots) {\n";
  os << "  __shared__ uint32_t s_count;\n  if (threadIdx.x == 0) s_count = 0;\n  __syncthreads();\n";
  os << "  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += gridDim.x * blockDim.x) {\n";
  int out_col = 0;
  std::vector<std::string> proj_vars;
  bool wrote_output = false;
  for (size_t i = 1; i < p.ops.size(); ++i) {
    const Rel& op = *p.ops[i];
    switch (op.kind) {
      case RelKind::Filter: {
        ASTER_ASSIGN_OR_RETURN(std::string pred, ExprToCuda(*op.predicate, inputs, "row"));
        os << "    if (!(" << pred << ")) continue;\n";
        break;
      }
      case RelKind::Project: {
        proj_vars.clear();
        for (size_t k = 0; k < op.exprs.size(); ++k) {
          ASTER_ASSIGN_OR_RETURN(std::string e, ExprToCuda(*op.exprs[k], inputs, "row"));
          std::string var = "p" + std::to_string(i) + "_" + std::to_string(k);
          os << "    " << CudaType(op.output.fields[k].type) << " " << var << " = " << e << ";\n";
          proj_vars.push_back(var);
        }
        break;
      }
      case RelKind::Aggregate: {
        // Partial aggregate into a global open addressing table keyed by hash of group keys.
        spec.has_partial_aggregate = true;
        os << "    uint64_t gh = 0x51ed270b27f1f4c3ull;\n";
        for (const auto& k : op.group_keys) {
          ASTER_ASSIGN_OR_RETURN(std::string ke, ExprToCuda(*k, inputs, "row"));
          os << "    gh = hash64(gh ^ (uint64_t)(" << ke << "));\n";
        }
        os << "    uint32_t slot = (uint32_t)(gh & (table_slots - 1));\n";
        os << "    for (;;) { uint64_t prev = atomicCAS((unsigned long long*)&partial_keys[slot], 0ull, gh | 1ull); if (prev == 0ull || prev == (gh | 1ull)) break; slot = (slot + 1) & (table_slots - 1); }\n";
        for (size_t k = 0; k < op.aggregates.size(); ++k) {
          const auto& agg = op.aggregates[k];
          std::string val = "1.0";
          if (!agg.args.empty()) { ASTER_ASSIGN_OR_RETURN(val, ExprToCuda(*agg.args[0], inputs, "row")); }
          std::string cell = "partial_vals[slot * " + std::to_string(op.aggregates.size()) + " + " + std::to_string(k) + "]";
          if (agg.function == "sum" || agg.function == "avg") os << "    atomicAdd(&" << cell << ", (double)(" << val << "));\n";
          else if (agg.function == "min") os << "    { double v = (double)(" << val << "); unsigned long long* a = (unsigned long long*)&" << cell << "; unsigned long long o = *a, s; do { s = o; if (__longlong_as_double(s) <= v) break; o = atomicCAS(a, s, __double_as_longlong(v)); } while (o != s); }\n";
          else if (agg.function == "max") os << "    { double v = (double)(" << val << "); unsigned long long* a = (unsigned long long*)&" << cell << "; unsigned long long o = *a, s; do { s = o; if (__longlong_as_double(s) >= v) break; o = atomicCAS(a, s, __double_as_longlong(v)); } while (o != s); }\n";
          if (agg.function == "count" || agg.function == "count_star" || agg.function == "avg")
            os << "    atomicAdd(&partial_counts[slot * " << op.aggregates.size() << " + " << k << "], 1u);\n";
        }
        wrote_output = true;
        break;
      }
      case RelKind::Join: {
        spec.has_hash_probe = true;
        os << "    // hash probe: build table passed through partial_keys as (hash|1) -> row index in partial_counts\n";
        os << "    uint64_t jh = 0x9e3779b97f4a7c15ull;\n";
        for (int k : op.left_keys) {
          ColumnBinding b; b.index = k; b.type = op.output.fields.at(k).type;
          for (const auto& ib : inputs) if (ib.index == k) b = ib;
          os << "    jh = hash64(jh ^ (uint64_t)(" << DecodeSnippet(b, "row") << "));\n";
        }
        os << "    uint32_t jslot = (uint32_t)(jh & (table_slots - 1)); bool matched = false; uint32_t build_row = 0;\n";
        os << "    for (uint32_t probe = 0; probe < table_slots; ++probe) { uint64_t key = partial_keys[jslot]; if (key == 0ull) break; if (key == (jh | 1ull)) { matched = true; build_row = partial_counts[jslot]; break; } jslot = (jslot + 1) & (table_slots - 1); }\n";
        if (op.join_type == JoinType::Anti) os << "    if (matched) continue;\n";
        else os << "    if (!matched) continue;\n";
        break;
      }
      case RelKind::Limit: break;
      default: return Status::NotSupported("operator in fused pipeline");
    }
  }
  if (!wrote_output) {
    // Materialize survivors: row index plus projected values, compacted with a block level counter.
    os << "    uint32_t pos = atomicAdd(out.count, 1u);\n";
    os << "    ((uint32_t*)out.cols[0])[pos] = row;\n";
    for (size_t k = 0; k < proj_vars.size(); ++k) {
      const Rel& last = *p.ops.back();
      const Rel* proj = nullptr;
      for (size_t i = p.ops.size(); i-- > 1;) if (p.ops[i]->kind == RelKind::Project) { proj = p.ops[i].get(); break; }
      std::string t = proj ? CudaType(proj->output.fields[k].type) : CudaType(last.output.fields[k].type);
      os << "    ((" << t << "*)out.cols[" << k + 1 << "])[pos] = " << proj_vars[k] << ";\n";
    }
    spec.num_outputs = static_cast<uint32_t>(proj_vars.size() + 1);
    (void)out_col;
  }
  os << "  }\n}\n";
  spec.source = os.str();
  return spec;
}

}  // namespace aster::exec::fused
