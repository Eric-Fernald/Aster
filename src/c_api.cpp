#include "aster/c_api.h"

#include <cstring>
#include <sstream>

#include "aster/engine.hpp"
#include "aster/interop/dlpack_export.hpp"

#if ASTER_HAVE_ARROW
#include <arrow/c/bridge.h>
#include "aster/integration/arrow_bridge.hpp"
#endif

using namespace aster;

struct AsterEngine { std::unique_ptr<Engine> engine; };
struct AsterResult {
  exec::QueryResult result;
  RecordBatchPtr merged;
  std::vector<std::string> type_names;
};

namespace {
char* Dup(const std::string& s) {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}
void SetError(char** out, const Status& s) { if (out) *out = Dup(s.ToString()); }

// Minimal schema JSON: [{"name":"a","type":"i64"}, ...]
Result<Schema> ParseSchemaJson(const std::string& json) {
  Schema s;
  size_t pos = 0;
  auto find_str = [&](const std::string& key, size_t from, std::string* out, size_t* end) -> bool {
    size_t k = json.find("\"" + key + "\"", from);
    if (k == std::string::npos) return false;
    size_t q1 = json.find('"', json.find(':', k) + 1);
    size_t q2 = json.find('"', q1 + 1);
    *out = json.substr(q1 + 1, q2 - q1 - 1);
    *end = q2;
    return true;
  };
  while (true) {
    size_t obj = json.find('{', pos);
    if (obj == std::string::npos) break;
    size_t close = json.find('}', obj);
    std::string name, type, e;
    size_t end;
    if (!find_str("name", obj, &name, &end) || end > close) break;
    if (!find_str("type", obj, &type, &end) || end > close) return Status::Invalid("field without type");
    DataType t;
    static const std::pair<const char*, TypeId> map[] = {{"bool", TypeId::Bool}, {"i8", TypeId::Int8}, {"i16", TypeId::Int16}, {"i32", TypeId::Int32}, {"i64", TypeId::Int64},
        {"u8", TypeId::UInt8}, {"u16", TypeId::UInt16}, {"u32", TypeId::UInt32}, {"u64", TypeId::UInt64}, {"f32", TypeId::Float32}, {"f64", TypeId::Float64},
        {"date32", TypeId::Date32}, {"timestamp", TypeId::Timestamp}, {"string", TypeId::String}, {"binary", TypeId::Binary}};
    bool found = false;
    for (const auto& [n, id] : map) if (type == n) { t = DataType::Of(id); found = true; }
    if (!found && type.rfind("vector", 0) == 0) { t = DataType::Vector(static_cast<uint32_t>(std::atoi(type.c_str() + 7))); found = true; }
    if (!found && type.rfind("decimal", 0) == 0) { unsigned p = 18, sc = 2; std::sscanf(type.c_str(), "decimal(%u,%u)", &p, &sc); t = DataType::Decimal(p, sc); found = true; }
    if (!found) return Status::Invalid("unknown type " + type);
    s.fields.push_back({name, t, true});
    pos = close + 1;
  }
  return s;
}
}  // namespace

extern "C" {

const char* aster_version(void) { return "0.1.0"; }

AsterEngine* aster_engine_open(int mode, const char* data_dir, int device_id, char** error_out) {
  EngineConfig cfg;
  cfg.mode = mode == 2 ? HardwareMode::Coherent : mode == 1 ? HardwareMode::Discrete : HardwareMode::CpuOnly;
  cfg.device_ids = {device_id};
  if (data_dir && *data_dir) {
    std::string root = data_dir;
    cfg.data_dir = root + "/data"; cfg.wal_dir = root + "/wal"; cfg.nvme_spill_dir = root + "/spill";
    cfg.kernel_cache_dir = root + "/kernels"; cfg.bandwidth_cache_path = root + "/bandwidth.tsv";
  }
  auto e = Engine::Open(cfg);
  if (!e.ok()) { SetError(error_out, e.status()); return nullptr; }
  return new AsterEngine{std::move(e.value())};
}

void aster_engine_close(AsterEngine* e) { delete e; }
void aster_free_string(char* s) { std::free(s); }

int aster_create_table_json(AsterEngine* e, const char* name, const char* schema_json, char** error_out) {
  auto schema = ParseSchemaJson(schema_json ? schema_json : "");
  if (!schema.ok()) { SetError(error_out, schema.status()); return -1; }
  storage::TableInfo info;
  info.name = name;
  info.schema = schema.value();
  Status s = e->engine->CreateTable(info);
  if (!s.ok()) { SetError(error_out, s); return -1; }
  return 0;
}

int aster_import_parquet(AsterEngine* e, const char* table, const char* path, char** error_out) {
  Status s = e->engine->ImportParquet(table, path);
  if (!s.ok()) { SetError(error_out, s); return -1; }
  return 0;
}

int aster_compact(AsterEngine* e, const char* table, char** error_out) {
  Status s = e->engine->Compact(table);
  if (!s.ok()) { SetError(error_out, s); return -1; }
  return 0;
}

static AsterResult* Wrap(Result<exec::QueryResult> r, char** error_out) {
  if (!r.ok()) { SetError(error_out, r.status()); return nullptr; }
  auto* out = new AsterResult{std::move(r.value()), nullptr, {}};
  out->merged = out->result.Concat();
  out->merged->schema = out->result.schema;
  for (const auto& f : out->result.schema.fields) out->type_names.push_back(TypeToString(f.type));
  return out;
}

AsterResult* aster_query_substrait(AsterEngine* e, const uint8_t* plan, size_t len, int keep_on_device, char** error_out) {
  return Wrap(e->engine->QuerySubstrait(std::string(reinterpret_cast<const char*>(plan), len), keep_on_device != 0), error_out);
}

AsterResult* aster_query_substrait_json(AsterEngine* e, const char* json, int keep_on_device, char** error_out) {
  return Wrap(e->engine->QuerySubstraitJson(json ? json : "", keep_on_device != 0), error_out);
}

void aster_result_free(AsterResult* r) { delete r; }
int64_t aster_result_num_rows(const AsterResult* r) { return r->merged->num_rows(); }
int aster_result_num_columns(const AsterResult* r) { return static_cast<int>(r->result.schema.fields.size()); }
const char* aster_result_column_name(const AsterResult* r, int i) { return r->result.schema.fields.at(i).name.c_str(); }
const char* aster_result_column_type(const AsterResult* r, int i) { return r->type_names.at(i).c_str(); }
char* aster_result_metrics_json(const AsterResult* r) { return Dup(r->result.metrics.ToJson()); }

DLManagedTensor* aster_result_column_dlpack(const AsterResult* r, int i, char** error_out) {
  if (i < 0 || i >= static_cast<int>(r->merged->columns.size())) { SetError(error_out, Status::Invalid("column index")); return nullptr; }
  Column c = r->merged->columns[i];
  if (c.is_dictionary_encoded()) { SetError(error_out, Status::NotSupported("string column; use arrow export")); return nullptr; }
  auto t = interop::DlpackExporter::Export(c);
  if (!t.ok()) { SetError(error_out, t.status()); return nullptr; }
  return t.value();
}

int aster_result_column_arrow(const AsterResult* r, int i, void* arrow_array, void* arrow_schema, char** error_out) {
#if ASTER_HAVE_ARROW
  auto arr = integration::ToArrow(r->merged->columns.at(i));
  if (!arr.ok()) { SetError(error_out, arr.status()); return -1; }
  auto st = arrow::ExportArray(*arr.value(), static_cast<ArrowArray*>(arrow_array), static_cast<ArrowSchema*>(arrow_schema));
  if (!st.ok()) { SetError(error_out, Status::Internal(st.ToString())); return -1; }
  return 0;
#else
  (void)r; (void)i; (void)arrow_array; (void)arrow_schema;
  SetError(error_out, Status::NotSupported("built without Arrow"));
  return -1;
#endif
}

char* aster_result_to_tsv(const AsterResult* r, int64_t max_rows) {
  std::ostringstream os;
  const RecordBatch& b = *r->merged;
  for (size_t c = 0; c < b.schema.fields.size(); ++c) os << (c ? "\t" : "") << b.schema.fields[c].name;
  os << "\n";
  int64_t n = max_rows < 0 ? b.num_rows() : std::min<int64_t>(max_rows, b.num_rows());
  for (int64_t i = 0; i < n; ++i) {
    for (size_t c = 0; c < b.columns.size(); ++c) {
      const Column& col = b.columns[c];
      os << (c ? "\t" : "");
      if (!col.IsValid(i)) { os << "NULL"; continue; }
      switch (col.type.id) {
        case TypeId::String: case TypeId::Binary: os << col.GetString(i); break;
        case TypeId::Float32: os << col.Values<float>()[i]; break;
        case TypeId::Float64: os << col.Values<double>()[i]; break;
        case TypeId::Int32: case TypeId::Date32: os << col.Values<int32_t>()[i]; break;
        case TypeId::Int64: case TypeId::Timestamp: case TypeId::Decimal64: os << col.Values<int64_t>()[i]; break;
        case TypeId::Bool: case TypeId::UInt8: os << int(col.Values<uint8_t>()[i]); break;
        default: os << "?"; break;
      }
    }
    os << "\n";
  }
  return Dup(os.str());
}

char* aster_capabilities_tsv(AsterEngine* e) {
  std::ostringstream os;
  for (const auto& en : e->engine->registry().Entries()) os << en.key << "\t" << integration::SupportName(en.support) << "\t" << en.note << "\n";
  return Dup(os.str());
}

char* aster_bandwidth_table(AsterEngine* e) { return Dup(e->engine->memory().bandwidth().ToString()); }

}  // extern "C"
