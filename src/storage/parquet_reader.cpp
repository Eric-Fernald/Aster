#include "aster/storage/parquet_reader.hpp"

#include "aster/integration/arrow_bridge.hpp"

#if ASTER_HAVE_ARROW
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#endif

namespace aster::storage {

bool ParquetReader::Available() {
#if ASTER_HAVE_ARROW
  return true;
#else
  return false;
#endif
}

#if ASTER_HAVE_ARROW
namespace {
Result<std::unique_ptr<parquet::arrow::FileReader>> OpenReader(const std::string& path) {
  auto infile = arrow::io::ReadableFile::Open(path);
  if (!infile.ok()) return Status::IoError(infile.status().ToString());
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto st = parquet::arrow::OpenFile(*infile, arrow::default_memory_pool(), &reader);
  if (!st.ok()) return Status::IoError(st.ToString());
  reader->set_use_threads(true);
  return reader;
}
}  // namespace

Result<Schema> ParquetReader::ReadSchema(const std::string& path) {
  ASTER_ASSIGN_OR_RETURN(auto reader, OpenReader(path));
  std::shared_ptr<arrow::Schema> schema;
  auto st = reader->GetSchema(&schema);
  if (!st.ok()) return Status::IoError(st.ToString());
  return integration::FromArrow(schema);
}

Result<int64_t> ParquetReader::RowCount(const std::string& path) {
  ASTER_ASSIGN_OR_RETURN(auto reader, OpenReader(path));
  return reader->parquet_reader()->metadata()->num_rows();
}

Result<std::vector<RecordBatchPtr>> ParquetReader::ReadFile(const std::string& path, const ParquetReadOptions& opts) {
  ASTER_ASSIGN_OR_RETURN(auto reader, OpenReader(path));
  reader->set_batch_size(opts.batch_rows);
  std::shared_ptr<arrow::Schema> schema;
  auto st = reader->GetSchema(&schema);
  if (!st.ok()) return Status::IoError(st.ToString());
  std::vector<int> indices;
  for (const auto& name : opts.columns) {
    int i = schema->GetFieldIndex(name);
    if (i < 0) return Status::NotFound("parquet column " + name);
    indices.push_back(i);
  }
  std::unique_ptr<arrow::RecordBatchReader> rb;
  st = opts.columns.empty() ? reader->GetRecordBatchReader(&rb) : reader->GetRecordBatchReader(indices, &rb);
  if (!st.ok()) return Status::IoError(st.ToString());
  std::vector<RecordBatchPtr> out;
  for (;;) {
    std::shared_ptr<arrow::RecordBatch> b;
    st = rb->ReadNext(&b);
    if (!st.ok()) return Status::IoError(st.ToString());
    if (!b) break;
    ASTER_ASSIGN_OR_RETURN(auto converted, integration::FromArrow(b));
    out.push_back(converted);
  }
  return out;
}
#else
Result<Schema> ParquetReader::ReadSchema(const std::string&) { return Status::NotSupported("built without Arrow/Parquet"); }
Result<int64_t> ParquetReader::RowCount(const std::string&) { return Status::NotSupported("built without Arrow/Parquet"); }
Result<std::vector<RecordBatchPtr>> ParquetReader::ReadFile(const std::string&, const ParquetReadOptions&) {
  return Status::NotSupported("built without Arrow/Parquet");
}
#endif

}  // namespace aster::storage
