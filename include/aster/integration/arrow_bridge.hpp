#pragma once
#include <memory>

#include "aster/common/column.hpp"
#include "aster/common/status.hpp"

#if ASTER_HAVE_ARROW
#include <arrow/api.h>
#endif

namespace aster::integration {

bool ArrowAvailable();

#if ASTER_HAVE_ARROW
// Zero copy where layouts match (fixed width, utf8, validity); dictionary arrays map to encoded columns.
Result<Column> FromArrow(const std::shared_ptr<arrow::Array>& arr);
Result<std::shared_ptr<arrow::Array>> ToArrow(const Column& col);
Result<RecordBatchPtr> FromArrow(const std::shared_ptr<arrow::RecordBatch>& batch);
Result<std::shared_ptr<arrow::RecordBatch>> ToArrow(const RecordBatch& batch);
Result<Schema> FromArrow(const std::shared_ptr<arrow::Schema>& schema);
Result<std::shared_ptr<arrow::Schema>> ToArrow(const Schema& schema);
Result<DataType> FromArrow(const arrow::DataType& t);
Result<std::shared_ptr<arrow::DataType>> ToArrow(const DataType& t);
#endif

}  // namespace aster::integration
