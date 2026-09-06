# Roadmap and status

Each phase has an exit criterion measured by `aster-bench`. Do not start the next phase until the
criterion is met. Status reflects what the code in this repository implements today; performance
exits need the target hardware.

## Phase 0: Spike

| Item | Status |
|---|---|
| DuckDB extension intercepting Substrait, filter + aggregate, Arrow back | `extensions/duckdb/aster_extension.cpp` (table function over `get_substrait`) |
| Benchmark harness with bytes moved and kernel time from the start | `bench/harness`, `aster-bench` |
| Tier bandwidth microbenchmark | `aster-bwprobe`, `memory::ProbeBandwidth` |

Exit: TPC-H Q1 and Q6 end to end, cost per query reported. Both run through `Engine::Query`
(tests `engine_q1_*`, `engine_q6_*`); the GPU number needs an H100 class node.

## Phase 1: Single GPU, fully resident

| Item | Status |
|---|---|
| Full TPC-H operator coverage via libcudf | `exec::CudfBackend` (join, groupby, sort, concat); CPU reference for everything |
| Capability registry and per subtree fallback | `integration::CapabilityRegistry`, `FallbackRouter` |
| Native segment writer/reader | `storage::SegmentWriter/Reader` |
| RMM pool allocator, arena per query | `memory::PoolAllocator` (size classes, per query arenas) |

Exit: 22 queries at SF100 resident, 5x cost efficiency vs DuckDB. Eight queries are expressed in
IR (`bench/harness/tpch.cpp`); all 22 are in `bench/tpch/queries.sql` for the DuckDB host path.

## Phase 2: Tiered memory manager

| Item | Status |
|---|---|
| Column chunk pages, residency map, prefetch, eviction | `memory::MemoryManager`, `ResidencyMap`, `Prefetcher`, `EvictionPolicy` |
| GPUDirect Storage path | `memory::CuFileReader` |
| Spill controller for hash tables and sort runs | `memory::SpillController`, partitioned `HashJoinBuild`, `SortAccumulator` runs |
| Coherent mode prototype | `MemoryManager` coherent mode (managed memory + advise/prefetch hints) |

Exit: SF1000 at 5 to 10x VRAM within 3x of SF100 per row. Sweep with `--vram-mb`.

## Phase 3: Compressed execution and fusion

| Item | Status |
|---|---|
| Dictionary, FOR, RLE in the segment format | `storage/encodings` |
| Predicate rewrite into the encoded domain | `planner::PredicateRewriter` |
| NVRTC pipeline fusion with kernel cache | `exec::fused::Codegen`, `JitCompiler`, `KernelCache`, `FusedPipelineRunner` |
| Late materialization | selection vectors through `TileScheduler`; codegen decodes payload only for survivors |

Exit: bytes moved per query down 3x vs phase 2; fused filter to aggregate 2x faster than libcudf chains.

## Phase 4: Ingest and pruning

| Item | Status |
|---|---|
| Delta store, WAL, compactor | `ingest/` |
| Zone maps and bloom filters written by compactor, used by planner | `storage::ZoneMap`, `BloomFilter`, `planner::SegmentPruner` |
| ClickBench full run | 14 query subset in IR, full 43 in `bench/clickbench/queries.sql` |

Exit: 1 GB/s append while ClickBench runs (`aster-bench --ingest`); 5x cost efficiency vs DuckDB.

## Phase 5: Multi GPU

| Item | Status |
|---|---|
| Partitioned load, copartitioned local joins | `exchange::HashPartitioner`, `ResidencyDirectory::RegisterPartition` |
| NCCL exchange operator | `exchange::NcclTransport`, `ExchangeOperator` (broadcast smaller side on PCIe) |
| Shared residency map and GPU aware placement | `exchange::ResidencyDirectory::BestGpuFor` |

Exit: SF3000 on 8 GPUs with near linear scaling on copartitioned joins.

## Phase 6: ML interop and hardening

| Item | Status |
|---|---|
| DLPack export, Python client | `interop::DlpackExporter`, `python/aster` |
| cuVS vector index and ANN operators | `exec::CagraIndex`, `BruteForceIndex`, `vector_distance` |
| Cross node exchange via UCX | `exchange::UcxTransport` |
| Chaos testing | `durability::FaultInjector`, `tests/test_ingest.cpp` chaos cases, `scripts/chaos_test.sh` |

## Open questions from the plan

1. HMM page fault performance on non Grace hardware: `MemoryManager` treats coherent mode as
   available whenever the device reports pageable memory access; measure with `aster-bwprobe`.
2. Encoded domain rewrite in DuckDB vs our planner: done in our planner per chunk, so it stays
   decoupled from DuckDB internals.
3. Blackwell decompression through nvCOMP: `storage::CompressGeneral` is the seam.
4. Partition key selection for multi GPU: `TableInfo::partition_keys` is declared by the user for now.
5. Delta store queryable on GPU: pinned batches ship per query (`SegmentScanSource` unions them);
   measure the crossover with `--ingest`.
