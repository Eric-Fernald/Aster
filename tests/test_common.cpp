#include "aster/common/column.hpp"
#include "aster/common/hash.hpp"
#include "aster/common/status.hpp"
#include "test_framework.hpp"

using namespace aster;

ASTER_TEST(status_result_roundtrip) {
  Result<int> r = 5;
  ASTER_CHECK(r.ok());
  ASTER_CHECK_EQ(r.value(), 5);
  Result<int> e = Status::NotFound("x");
  ASTER_CHECK(!e.ok());
  ASTER_CHECK_EQ(e.status().code() == ErrorCode::NotFound, true);
}

ASTER_TEST(crc32c_known_vector) {
  ASTER_CHECK_EQ(Crc32c("123456789", 9), 0xE3069283u);
  ASTER_CHECK_EQ(Crc32c("", 0), 0u);
}

ASTER_TEST(hash64_stable_and_distinct) {
  uint64_t a = Hash64("hello"), b = Hash64("hello"), c = Hash64("hellp");
  ASTER_CHECK_EQ(a, b);
  ASTER_CHECK(a != c);
  std::string big(1000, 'x');
  ASTER_CHECK(Hash64(big) != Hash64(big + "y"));
}

ASTER_TEST(column_string_and_validity) {
  std::vector<bool> valid = {true, false, true};
  Column c = MakeStringColumn({"a", "", "ccc"}, &valid);
  ASTER_CHECK_EQ(c.length, 3);
  ASTER_CHECK_EQ(c.null_count, 1);
  ASTER_CHECK(!c.IsValid(1));
  ASTER_CHECK_EQ(std::string(c.GetString(2)), "ccc");
}

ASTER_TEST(column_take_and_concat) {
  Column ints = MakeColumn<int64_t>(TypeId::Int64, {10, 20, 30, 40});
  Column taken = TakeColumn(ints, std::vector<RowIdx>{3, 1});
  ASTER_CHECK_EQ(taken.Values<int64_t>()[0], 40);
  ASTER_CHECK_EQ(taken.Values<int64_t>()[1], 20);
  auto b1 = std::make_shared<RecordBatch>();
  b1->schema.fields = {{"s", DataType::Of(TypeId::String), true}};
  b1->columns = {MakeStringColumn({"x", "yy"})};
  auto b2 = std::make_shared<RecordBatch>();
  b2->schema = b1->schema;
  b2->columns = {MakeStringColumn({"zzz"})};
  auto all = ConcatBatches({b1, b2});
  ASTER_CHECK_EQ(all->num_rows(), 3);
  ASTER_CHECK_EQ(std::string(all->columns[0].GetString(2)), "zzz");
}

ASTER_TEST(dictionary_column_decode) {
  Column d = MakeDictionaryColumn({"EU", "US"}, {1, 0, 1}, 77);
  ASTER_CHECK(d.is_dictionary_encoded());
  ASTER_CHECK_EQ(std::string(d.GetString(0)), "US");
  Column plain = d.DecodeDictionary();
  ASTER_CHECK(!plain.is_dictionary_encoded());
  ASTER_CHECK_EQ(std::string(plain.GetString(1)), "EU");
}
