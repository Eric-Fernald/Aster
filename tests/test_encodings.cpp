#include "aster/common/hash.hpp"
#include "aster/storage/bloom_filter.hpp"
#include "aster/storage/encoding.hpp"
#include "aster/storage/encoding_selector.hpp"
#include "aster/storage/zone_map.hpp"
#include "test_framework.hpp"

using namespace aster;
using namespace aster::storage;

namespace {
Column RoundTrip(const Column& c, Encoding e, Codec codec = Codec::None) {
  EncodeOptions o;
  o.encoding = e;
  o.outer_codec = codec;
  auto enc = EncodeColumn(c, o);
  if (!enc.ok()) throw aster_test::Failure{"encode: " + enc.status().ToString()};
  auto dec = DecodeChunk(enc.value());
  if (!dec.ok()) throw aster_test::Failure{"decode: " + dec.status().ToString()};
  return dec.value();
}
}  // namespace

ASTER_TEST(bitpack_roundtrip) {
  std::vector<uint64_t> v = {0, 1, 5, 7, 3, 6};
  std::vector<uint8_t> out(PackedBytes(6, 3));
  PackBits(v.data(), 6, 3, out.data());
  for (uint32_t i = 0; i < 6; ++i) ASTER_CHECK_EQ(UnpackBit(out.data(), i, 3), v[i]);
  ASTER_CHECK_EQ(int(BitsNeeded(7)), 3);
  ASTER_CHECK_EQ(int(BitsNeeded(0)), 0);
}

ASTER_TEST(plain_roundtrip_with_nulls) {
  std::vector<bool> valid = {true, false, true};
  Column c = MakeColumn<int32_t>(TypeId::Int32, {1, 2, 3}, &valid);
  Column d = RoundTrip(c, Encoding::Plain);
  ASTER_CHECK_EQ(d.length, 3);
  ASTER_CHECK_EQ(d.null_count, 1);
  ASTER_CHECK(!d.IsValid(1));
  ASTER_CHECK_EQ(d.Values<int32_t>()[2], 3);
  Column s = RoundTrip(MakeStringColumn({"ab", "", "xyz"}), Encoding::Plain);
  ASTER_CHECK_EQ(std::string(s.GetString(2)), "xyz");
}

ASTER_TEST(dictionary_roundtrip_stays_encoded) {
  Column c = MakeStringColumn({"EU", "US", "EU", "ASIA", "EU"});
  EncodeOptions o; o.encoding = Encoding::Dictionary;
  ASTER_ASSIGN_OK(auto enc, EncodeColumn(c, o));
  ASTER_ASSIGN_OK(auto view, ViewDictionary(enc.data.data(), enc.data.size()));
  ASTER_CHECK_EQ(view.values.size(), size_t(3));
  ASTER_CHECK_EQ(view.LookupCode("ASIA"), 2);
  ASTER_CHECK_EQ(view.LookupCode("MARS"), -1);
  ASTER_CHECK_EQ(view.CodeAt(3), 2);
  ASTER_ASSIGN_OK(Column d, DecodeChunk(enc));
  ASTER_CHECK(d.is_dictionary_encoded());
  ASTER_CHECK_EQ(std::string(d.GetString(3)), "ASIA");
  ASTER_CHECK(enc.bytes() < sizeof(ChunkHeader) + 8 + 4 * 4 + 6 + 5 + 4);
}

ASTER_TEST(dictionary_int_roundtrip) {
  Column c = MakeColumn<int64_t>(TypeId::Int64, {100, 200, 100, 100});
  Column d = RoundTrip(c, Encoding::Dictionary);
  ASTER_CHECK(!d.is_dictionary_encoded());
  ASTER_CHECK_EQ(d.Values<int64_t>()[1], 200);
}

ASTER_TEST(for_bitpack_roundtrip_and_view) {
  Column c = MakeColumn<int64_t>(TypeId::Int64, {1000, 1003, 1001, 1007});
  EncodeOptions o; o.encoding = Encoding::ForBitPack;
  ASTER_ASSIGN_OK(auto enc, EncodeColumn(c, o));
  ASTER_ASSIGN_OK(auto view, ViewFor(enc.data.data(), enc.data.size()));
  ASTER_CHECK_EQ(view.reference, 1000);
  ASTER_CHECK_EQ(int(view.bit_width), 3);
  ASTER_CHECK_EQ(view.ValueAt(3), 1007);
  ASTER_CHECK(enc.bytes() < 4 * 8 + sizeof(ChunkHeader) + 16);
  ASTER_ASSIGN_OK(Column d, DecodeChunk(enc));
  ASTER_CHECK_EQ(d.Values<int64_t>()[2], 1001);
  std::vector<bool> valid = {true, false, true};
  Column n = MakeColumn<int32_t>(TypeId::Int32, {5, 0, 9}, &valid);
  Column dn = RoundTrip(n, Encoding::ForBitPack);
  ASTER_CHECK(!dn.IsValid(1));
  ASTER_CHECK_EQ(dn.Values<int32_t>()[2], 9);
}

ASTER_TEST(rle_roundtrip_and_runs) {
  Column c = MakeColumn<int32_t>(TypeId::Int32, {7, 7, 7, 8, 8, 9});
  EncodeOptions o; o.encoding = Encoding::RunLength;
  ASTER_ASSIGN_OK(auto enc, EncodeColumn(c, o));
  ASTER_ASSIGN_OK(auto runs, ViewRuns(enc.data.data(), enc.data.size()));
  ASTER_CHECK_EQ(runs.num_runs, 3u);
  ASTER_CHECK_EQ(runs.run_lengths[0], 3u);
  ASTER_CHECK_EQ(runs.ValueAt(1), 8);
  ASTER_ASSIGN_OK(Column d, DecodeChunk(enc));
  ASTER_CHECK_EQ(d.Values<int32_t>()[5], 9);
}

ASTER_TEST(delta_roundtrip) {
  Column c = MakeColumn<int64_t>(TypeId::Int64, {100, 101, 102, 104, 105});
  Column d = RoundTrip(c, Encoding::Delta);
  for (int i = 0; i < 5; ++i) ASTER_CHECK_EQ(d.Values<int64_t>()[i], c.Values<int64_t>()[i]);
  Column ts = MakeColumn<int64_t>(TypeId::Timestamp, {1'000'000, 1'000'010, 1'000'020});
  Column dts = RoundTrip(ts, Encoding::Delta);
  ASTER_CHECK(dts.type.id == TypeId::Timestamp);
  ASTER_CHECK_EQ(dts.Values<int64_t>()[2], 1'000'020);
}

ASTER_TEST(general_codec_none_wraps_inner) {
  Column c = MakeColumn<int32_t>(TypeId::Int32, {1, 1, 1, 2});
  EncodeOptions o; o.encoding = Encoding::RunLength; o.outer_codec = Codec::None;
  ASTER_ASSIGN_OK(auto enc, EncodeColumn(c, o));
  ASTER_CHECK(enc.encoding == Encoding::RunLength);
  ASTER_CHECK_EQ(CodecAvailable(Codec::None), true);
}

ASTER_TEST(zone_map_and_hll) {
  Column c = MakeColumn<int64_t>(TypeId::Int64, {5, 1, 9, 3});
  ZoneMap z = ZoneMap::Compute(c);
  ASTER_CHECK(z.has_bounds);
  ASTER_CHECK_EQ(z.min_i64, 1);
  ASTER_CHECK_EQ(z.max_i64, 9);
  ASTER_CHECK(!z.sorted_asc);
  ASTER_CHECK(z.MayContainI64(4));
  ASTER_CHECK(!z.MayContainI64(10));
  ASTER_CHECK_EQ(z.distinct_estimate, 4u);
  std::vector<uint8_t> buf;
  z.Serialize(buf);
  const uint8_t* p = buf.data();
  ZoneMap back = ZoneMap::Deserialize(p, p + buf.size());
  ASTER_CHECK_EQ(back.max_i64, 9);
  HyperLogLog h(12);
  for (uint64_t i = 0; i < 100000; ++i) h.Add(HashInt(i));
  double est = double(h.Estimate());
  ASTER_CHECK(est > 90000 && est < 110000);
}

ASTER_TEST(zone_map_strings) {
  ZoneMap z = ZoneMap::Compute(MakeStringColumn({"banana", "apple", "cherry"}));
  ASTER_CHECK_EQ(z.min_str, "apple");
  ASTER_CHECK_EQ(z.max_str, "cherry");
  ASTER_CHECK(z.MayContainStr("blueberry"));
  ASTER_CHECK(!z.MayContainStr("zebra"));
}

ASTER_TEST(bloom_filter_membership) {
  Column c = MakeStringColumn({"alpha", "beta", "gamma"});
  BloomFilter b = BloomFilter::Build(c, 0.01);
  ASTER_CHECK(b.MayContain("alpha"));
  ASTER_CHECK(b.MayContain("gamma"));
  int fp = 0;
  for (int i = 0; i < 1000; ++i) if (b.MayContain("nope" + std::to_string(i))) ++fp;
  ASTER_CHECK(fp < 50);
  std::vector<uint8_t> buf;
  b.Serialize(buf);
  const uint8_t* p = buf.data();
  BloomFilter back = BloomFilter::Deserialize(p, p + buf.size());
  ASTER_CHECK(back.MayContain("beta"));
}

ASTER_TEST(encoding_selector_priorities) {
  ASTER_CHECK(SelectEncoding(MakeStringColumn(std::vector<std::string>(100, "same"))) == Encoding::Dictionary);
  std::vector<int64_t> runs(100, 3); for (int i = 50; i < 100; ++i) runs[i] = 4;
  ASTER_CHECK(SelectEncoding(MakeColumn<int64_t>(TypeId::Int64, runs)) == Encoding::RunLength);
  std::vector<int64_t> mono(100); for (int i = 0; i < 100; ++i) mono[i] = 1'000'000 + i * 3;
  ASTER_CHECK(SelectEncoding(MakeColumn<int64_t>(TypeId::Int64, mono)) == Encoding::Delta);
  std::vector<int64_t> narrow(100); for (int i = 0; i < 100; ++i) narrow[i] = 5000 + (i * 37) % 200;
  ASTER_CHECK(SelectEncoding(MakeColumn<int64_t>(TypeId::Int64, narrow)) == Encoding::ForBitPack);
  std::vector<double> f(100); for (int i = 0; i < 100; ++i) f[i] = i * 0.5;
  ASTER_CHECK(SelectEncoding(MakeColumn<double>(TypeId::Float64, f)) == Encoding::Plain);
}
