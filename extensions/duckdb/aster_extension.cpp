// DuckDB loadable extension: DuckDB parses and optimizes, the substrait extension serializes the
// physical plan, Aster executes it and hands Arrow layout chunks back through a table function.
//
//   LOAD 'aster';
//   SELECT * FROM aster_query('SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1');
//   SELECT aster_import_parquet('lineitem', '/data/lineitem.parquet');
//   SELECT * FROM aster_capabilities();
#define DUCKDB_EXTENSION_MAIN
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension_util.hpp"

#include <memory>
#include <mutex>

#include "aster/engine.hpp"

namespace duckdb {

namespace {

std::unique_ptr<aster::Engine>& GlobalEngine() {
  static std::unique_ptr<aster::Engine> engine;
  return engine;
}
std::mutex& EngineMutex() { static std::mutex m; return m; }

aster::Engine& EnsureEngine(ClientContext& context) {
  std::lock_guard<std::mutex> lk(EngineMutex());
  auto& e = GlobalEngine();
  if (!e) {
    aster::EngineConfig cfg;
    Value mode;
    if (context.TryGetCurrentSetting("aster_mode", mode)) {
      std::string m = mode.ToString();
      cfg.mode = m == "coherent" ? aster::HardwareMode::Coherent : m == "discrete" ? aster::HardwareMode::Discrete : aster::HardwareMode::CpuOnly;
    } else {
      cfg.mode = aster::HardwareMode::Discrete;
    }
    Value root;
    if (context.TryGetCurrentSetting("aster_root", root)) {
      std::string r = root.ToString();
      cfg.data_dir = r + "/data"; cfg.wal_dir = r + "/wal"; cfg.nvme_spill_dir = r + "/spill";
      cfg.kernel_cache_dir = r + "/kernels"; cfg.bandwidth_cache_path = r + "/bandwidth.tsv";
    }
    auto opened = aster::Engine::Open(cfg);
    if (!opened.ok()) throw IOException("aster: " + opened.status().ToString());
    e = std::move(opened.value());
  }
  return *e;
}

LogicalType ToDuck(const aster::DataType& t) {
  using aster::TypeId;
  switch (t.id) {
    case TypeId::Bool: return LogicalType::BOOLEAN;
    case TypeId::Int8: return LogicalType::TINYINT;
    case TypeId::Int16: return LogicalType::SMALLINT;
    case TypeId::Int32: return LogicalType::INTEGER;
    case TypeId::Int64: return LogicalType::BIGINT;
    case TypeId::UInt8: return LogicalType::UTINYINT;
    case TypeId::UInt16: return LogicalType::USMALLINT;
    case TypeId::UInt32: return LogicalType::UINTEGER;
    case TypeId::UInt64: return LogicalType::UBIGINT;
    case TypeId::Float32: return LogicalType::FLOAT;
    case TypeId::Float64: return LogicalType::DOUBLE;
    case TypeId::Date32: return LogicalType::DATE;
    case TypeId::Timestamp: return LogicalType::TIMESTAMP;
    case TypeId::Decimal64: return LogicalType::DECIMAL(t.precision, t.width);
    case TypeId::String: return LogicalType::VARCHAR;
    case TypeId::Binary: return LogicalType::BLOB;
    default: return LogicalType::VARCHAR;
  }
}

// Substrait bytes come from DuckDB's own substrait extension so the host owns parsing and optimization.
std::string PlanFor(ClientContext& context, const std::string& sql) {
  Connection con(*context.db);
  auto res = con.Query("CALL get_substrait('" + StringUtil::Replace(sql, "'", "''") + "')");
  if (res->HasError()) throw InvalidInputException("aster: get_substrait failed (is the substrait extension loaded?): " + res->GetError());
  auto chunk = res->Fetch();
  if (!chunk || chunk->size() == 0) throw InvalidInputException("aster: empty substrait plan");
  return chunk->GetValue(0, 0).GetValueUnsafe<string_t>().GetString();
}

struct QueryBindData : public TableFunctionData {
  std::shared_ptr<aster::exec::QueryResult> result;
  aster::RecordBatchPtr merged;
};
struct QueryGlobalState : public GlobalTableFunctionState {
  idx_t offset = 0;
};

unique_ptr<FunctionData> QueryBind(ClientContext& context, TableFunctionBindInput& input, vector<LogicalType>& return_types, vector<string>& names) {
  auto sql = input.inputs[0].GetValue<string>();
  aster::Engine& engine = EnsureEngine(context);
  std::string plan = PlanFor(context, sql);
  auto r = engine.QuerySubstrait(plan, false);
  if (!r.ok()) throw InvalidInputException("aster: " + r.status().ToString());
  auto bind = make_uniq<QueryBindData>();
  bind->result = std::make_shared<aster::exec::QueryResult>(std::move(r.value()));
  bind->merged = bind->result->Concat();
  for (const auto& f : bind->result->schema.fields) { names.push_back(f.name); return_types.push_back(ToDuck(f.type)); }
  return std::move(bind);
}

unique_ptr<GlobalTableFunctionState> QueryInit(ClientContext&, TableFunctionInitInput&) { return make_uniq<QueryGlobalState>(); }

void QueryScan(ClientContext&, TableFunctionInput& data, DataChunk& output) {
  auto& bind = data.bind_data->Cast<QueryBindData>();
  auto& state = data.global_state->Cast<QueryGlobalState>();
  const aster::RecordBatch& b = *bind.merged;
  idx_t total = static_cast<idx_t>(b.num_rows());
  if (state.offset >= total) { output.SetCardinality(0); return; }
  idx_t n = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - state.offset);
  for (idx_t c = 0; c < b.columns.size(); ++c) {
    const aster::Column& col = b.columns[c];
    Vector& v = output.data[c];
    for (idx_t i = 0; i < n; ++i) {
      int64_t row = static_cast<int64_t>(state.offset + i);
      if (!col.IsValid(row)) { FlatVector::SetNull(v, i, true); continue; }
      switch (col.type.id) {
        case aster::TypeId::Bool: FlatVector::GetData<bool>(v)[i] = col.Values<uint8_t>()[row] != 0; break;
        case aster::TypeId::Int8: FlatVector::GetData<int8_t>(v)[i] = col.Values<int8_t>()[row]; break;
        case aster::TypeId::Int16: FlatVector::GetData<int16_t>(v)[i] = col.Values<int16_t>()[row]; break;
        case aster::TypeId::Int32: FlatVector::GetData<int32_t>(v)[i] = col.Values<int32_t>()[row]; break;
        case aster::TypeId::Date32: FlatVector::GetData<date_t>(v)[i] = date_t(col.Values<int32_t>()[row]); break;
        case aster::TypeId::Int64: FlatVector::GetData<int64_t>(v)[i] = col.Values<int64_t>()[row]; break;
        case aster::TypeId::Timestamp: FlatVector::GetData<timestamp_t>(v)[i] = timestamp_t(col.Values<int64_t>()[row]); break;
        case aster::TypeId::Decimal64: FlatVector::GetData<int64_t>(v)[i] = col.Values<int64_t>()[row]; break;
        case aster::TypeId::UInt8: FlatVector::GetData<uint8_t>(v)[i] = col.Values<uint8_t>()[row]; break;
        case aster::TypeId::UInt16: FlatVector::GetData<uint16_t>(v)[i] = col.Values<uint16_t>()[row]; break;
        case aster::TypeId::UInt32: FlatVector::GetData<uint32_t>(v)[i] = col.Values<uint32_t>()[row]; break;
        case aster::TypeId::UInt64: FlatVector::GetData<uint64_t>(v)[i] = col.Values<uint64_t>()[row]; break;
        case aster::TypeId::Float32: FlatVector::GetData<float>(v)[i] = col.Values<float>()[row]; break;
        case aster::TypeId::Float64: FlatVector::GetData<double>(v)[i] = col.Values<double>()[row]; break;
        case aster::TypeId::String: case aster::TypeId::Binary: {
          auto s = col.GetString(row);
          FlatVector::GetData<string_t>(v)[i] = StringVector::AddString(v, s.data(), s.size());
          break;
        }
        default: FlatVector::SetNull(v, i, true); break;
      }
    }
  }
  state.offset += n;
  output.SetCardinality(n);
}

struct CapBindData : public TableFunctionData { std::vector<aster::integration::CapabilityEntry> entries; };
struct CapState : public GlobalTableFunctionState { idx_t offset = 0; };

unique_ptr<FunctionData> CapBind(ClientContext& context, TableFunctionBindInput&, vector<LogicalType>& types, vector<string>& names) {
  auto bind = make_uniq<CapBindData>();
  bind->entries = EnsureEngine(context).registry().Entries();
  names = {"key", "support", "note"};
  types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
  return std::move(bind);
}
unique_ptr<GlobalTableFunctionState> CapInit(ClientContext&, TableFunctionInitInput&) { return make_uniq<CapState>(); }
void CapScan(ClientContext&, TableFunctionInput& data, DataChunk& output) {
  auto& bind = data.bind_data->Cast<CapBindData>();
  auto& st = data.global_state->Cast<CapState>();
  idx_t n = MinValue<idx_t>(STANDARD_VECTOR_SIZE, bind.entries.size() - st.offset);
  for (idx_t i = 0; i < n; ++i) {
    const auto& e = bind.entries[st.offset + i];
    output.SetValue(0, i, Value(e.key));
    output.SetValue(1, i, Value(aster::integration::SupportName(e.support)));
    output.SetValue(2, i, Value(e.note));
  }
  st.offset += n;
  output.SetCardinality(n);
}

void ImportParquetFn(DataChunk& args, ExpressionState& state, Vector& result) {
  auto& context = state.GetContext();
  BinaryExecutor::Execute<string_t, string_t, string_t>(args.data[0], args.data[1], result, args.size(), [&](string_t table, string_t path) {
    aster::Status s = EnsureEngine(context).ImportParquet(table.GetString(), path.GetString());
    return StringVector::AddString(result, s.ok() ? "ok" : s.ToString());
  });
}

void CompactFn(DataChunk& args, ExpressionState& state, Vector& result) {
  auto& context = state.GetContext();
  UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t table) {
    aster::Status s = EnsureEngine(context).Compact(table.GetString());
    return StringVector::AddString(result, s.ok() ? "ok" : s.ToString());
  });
}

}  // namespace

class AsterExtension : public Extension {
 public:
  void Load(DuckDB& db) override {
    auto& instance = *db.instance;
    auto& config = DBConfig::GetConfig(instance);
    config.AddExtensionOption("aster_mode", "Aster hardware mode: cpu, discrete or coherent", LogicalType::VARCHAR, Value("discrete"));
    config.AddExtensionOption("aster_root", "Aster data root directory", LogicalType::VARCHAR, Value("/var/tmp/aster"));
    TableFunction query("aster_query", {LogicalType::VARCHAR}, QueryScan, QueryBind, QueryInit);
    ExtensionUtil::RegisterFunction(instance, query);
    TableFunction caps("aster_capabilities", {}, CapScan, CapBind, CapInit);
    ExtensionUtil::RegisterFunction(instance, caps);
    ExtensionUtil::RegisterFunction(instance, ScalarFunction("aster_import_parquet", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, ImportParquetFn));
    ExtensionUtil::RegisterFunction(instance, ScalarFunction("aster_compact", {LogicalType::VARCHAR}, LogicalType::VARCHAR, CompactFn));
  }
  std::string Name() override { return "aster"; }
  std::string Version() const override { return "0.1.0"; }
};

}  // namespace duckdb

extern "C" {
DUCKDB_EXTENSION_API void aster_init(duckdb::DatabaseInstance& db) {
  duckdb::DuckDB wrapper(db);
  wrapper.LoadExtension<duckdb::AsterExtension>();
}
DUCKDB_EXTENSION_API const char* aster_version() { return duckdb::DuckDB::LibraryVersion(); }
}
