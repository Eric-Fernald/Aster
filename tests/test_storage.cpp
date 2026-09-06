#include <filesystem>
#include <fstream>

#include "aster/storage/catalog.hpp"
#include "aster/storage/parquet_reader.hpp"
#include "aster/storage/segment.hpp"
#include "test_framework.hpp"
#include "test_helpers.hpp"

using namespace aster;
using namespace aster::storage;

ASTER_TEST(segment_write_read_roundtrip) {
  std::string dir = aster_test::TempDir("seg");
  auto batch = aster_test::MakeLineitem(5000);
  SegmentWriteOptions opts;
  ASTER_ASSIGN_OK(SegmentMeta meta, SegmentWriter::Write(*batch, 1, dir + "/seg_1.aseg", "lineitem", opts));
  ASTER_CHECK_EQ(meta.num_rows, 5000u);
  ASTER_CHECK_EQ(meta.columns.size(), size_t(8));
  ASTER_CHECK(meta.columns[5].encoding == Encoding::Dictionary);   // l_returnflag, 3 distinct
  ASTER_CHECK(meta.columns[1].encoding == Encoding::ForBitPack);   // l_quantity 1..50
  ASTER_CHECK(meta.columns[0].zone.has_bounds);
  ASTER_CHECK(!meta.columns[5].bloom.empty());
  for (const auto& c : meta.columns) ASTER_CHECK_EQ(c.offset % kChunkAlignment, 0u);

  ASTER_ASSIGN_OK(SegmentMeta back, SegmentReader::ReadMeta(dir + "/seg_1.aseg"));
  ASTER_CHECK_EQ(back.segment_id, 1u);
  ASTER_CHECK_EQ(back.columns[5].name, "l_returnflag");
  ASTER_ASSIGN_OK(auto all, SegmentReader::ReadAll(back));
  ASTER_CHECK_EQ(all->num_rows(), 5000);
  for (int64_t i = 0; i < 5000; i += 997) {
    ASTER_CHECK_EQ(all->columns[0].Values<int64_t>()[i], batch->columns[0].Values<int64_t>()[i]);
    ASTER_CHECK_EQ(all->columns[1].Values<int32_t>()[i], batch->columns[1].Values<int32_t>()[i]);
    ASTER_CHECK_EQ(std::string(all->columns[5].GetString(i)), std::string(batch->columns[5].GetString(i)));
    ASTER_CHECK_NEAR(all->columns[2].Values<double>()[i], batch->columns[2].Values<double>()[i], 1e-9);
  }
  ASTER_CHECK(all->columns[5].is_dictionary_encoded());
  auto pages = back.Pages();
  ASTER_CHECK_EQ(pages.size(), size_t(8));
  ASTER_CHECK_EQ(pages[2].locator.bytes, back.columns[2].bytes);
}

ASTER_TEST(segment_checksum_detects_corruption) {
  std::string dir = aster_test::TempDir("corrupt");
  auto batch = aster_test::MakeLineitem(500);
  std::string path = dir + "/seg_2.aseg";
  ASTER_ASSIGN_OK(SegmentMeta meta, SegmentWriter::Write(*batch, 2, path, "lineitem", {}));
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(meta.columns[0].offset + 40));
    char junk = 'Z';
    f.write(&junk, 1);
  }
  auto col = SegmentReader::ReadColumn(meta, 0);
  ASTER_CHECK(!col.ok());
  ASTER_CHECK(col.status().code() == ErrorCode::Corrupt);
}

ASTER_TEST(catalog_manifest_and_atomic_swap) {
  std::string dir = aster_test::TempDir("catalog");
  Catalog cat(dir);
  ASTER_CHECK_OK(cat.Open());
  TableInfo info;
  info.name = "lineitem";
  info.schema = aster_test::LineitemSchema();
  ASTER_CHECK_OK(cat.CreateTable(info));
  auto batch = aster_test::MakeLineitem(1000);
  SegmentId id = cat.NextSegmentId();
  ASTER_ASSIGN_OK(SegmentMeta meta, SegmentWriter::Write(*batch, id, cat.SegmentPath("lineitem", id), "lineitem", {}));
  ASTER_CHECK_OK(cat.AppendSegment("lineitem", std::make_shared<SegmentMeta>(meta)));
  ASTER_ASSIGN_OK(auto snap1, cat.Snapshot("lineitem"));
  ASTER_CHECK_EQ(snap1->segments.size(), size_t(1));
  ASTER_CHECK_EQ(snap1->version, 1u);
  ASTER_CHECK_OK(cat.SwapSegments("lineitem", {}));
  ASTER_ASSIGN_OK(auto snap2, cat.Snapshot("lineitem"));
  ASTER_CHECK_EQ(snap2->segments.size(), size_t(0));
  ASTER_CHECK_EQ(snap1->segments.size(), size_t(1));  // old snapshot survives for in flight readers

  Catalog reopened(dir);
  ASTER_CHECK_OK(reopened.Open());
  ASTER_CHECK(reopened.HasTable("lineitem"));
  ASTER_ASSIGN_OK(auto info2, reopened.GetTable("lineitem"));
  ASTER_CHECK_EQ(info2.schema.fields.size(), size_t(8));
  ASTER_CHECK(reopened.NextSegmentId() > id);
}

ASTER_TEST(parquet_reader_reports_availability) {
  if (!ParquetReader::Available()) {
    auto r = ParquetReader::ReadFile("/nonexistent.parquet");
    ASTER_CHECK(!r.ok());
    ASTER_CHECK(r.status().code() == ErrorCode::NotSupported);
  }
}
