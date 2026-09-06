#include <cstdio>

#include "aster/storage/segment.hpp"

int main(int argc, char** argv) {
  using namespace aster;
  if (argc < 2) { std::fprintf(stderr, "usage: aster-segment-inspect FILE.aseg [--rows N]\n"); return 2; }
  int64_t rows = 0;
  for (int i = 2; i + 1 < argc; ++i) if (std::string(argv[i]) == "--rows") rows = std::atoll(argv[i + 1]);
  auto meta = storage::SegmentReader::ReadMeta(argv[1]);
  if (!meta.ok()) { std::fprintf(stderr, "%s\n", meta.status().ToString().c_str()); return 1; }
  const auto& m = meta.value();
  std::printf("segment %llu table=%s rows=%llu encoded_bytes=%llu tile_rows=%u columns=%zu\n", (unsigned long long)m.segment_id,
              m.table.c_str(), (unsigned long long)m.num_rows, (unsigned long long)m.encoded_bytes, m.tile_rows, m.columns.size());
  for (size_t i = 0; i < m.columns.size(); ++i) {
    const auto& c = m.columns[i];
    std::printf("  [%zu] %-20s %-10s enc=%-12s codec=%d offset=%llu bytes=%u crc=%08x nulls=%u bloom=%zuB dict=%llu\n      zone: %s\n", i,
                c.name.c_str(), TypeToString(c.type).c_str(), storage::EncodingName(c.encoding), int(c.codec), (unsigned long long)c.offset, c.bytes,
                c.checksum, c.null_count, c.bloom.bytes(), (unsigned long long)c.dictionary_id, c.zone.ToString().c_str());
  }
  if (rows > 0) {
    auto batch = storage::SegmentReader::ReadAll(m);
    if (!batch.ok()) { std::fprintf(stderr, "%s\n", batch.status().ToString().c_str()); return 1; }
    std::printf("%s", batch.value()->ToString(rows).c_str());
  }
  return 0;
}
