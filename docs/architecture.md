# Architecture

This document maps the design plan's components onto the code.

## Query data flow

1. The host emits a Substrait plan. `integration::SubstraitConsumer` turns it into the internal IR
   (`plan::Rel`, `plan::Expr`); `plan::PlanBuilder` builds the same IR directly for tests and tools.
2. `integration::FallbackRouter` checks every node against `CapabilityRegistry`. Nodes it cannot run
   on GPU are marked `Placement::Cpu`; a GPU capable subtree under a CPU parent is repriced with the
   measured HBM to host bandwidth and pulled to CPU when the round trip costs more than the speedup.
3. `planner::SegmentPruner` walks zone maps and bloom filters from the catalog snapshot before any
   data moves. Pruned segment and row counts flow into the query metrics.
4. `planner::PipelineSplitter` breaks the tree at aggregate finalize, sort, window and exchange, and
   turns each join build side into its own pipeline. Each pipeline gets a shape hash (operator
   sequence, types, encodings) that keys the kernel cache.
5. `planner::CostModel` estimates rows and bytes per tier from the residency map and
   `memory::BandwidthTable`; `planner::PlacementPolicy` picks resident, stream from host, stream from
   NVMe or CPU per pipeline; `planner::PrefetchScheduler` orders page fetches so the first tiles of
   every pipeline are in flight while kernels compile.
6. `exec::Executor` marks scheduled pages, enqueues prefetches, and runs `exec::TileScheduler`.
   Tiles of `tile_rows` rows flow from `SegmentScanSource` (pages acquired through the memory
   manager, decoded once per segment, dictionary columns kept as codes) through filter, project,
   probe and limit into the breaker state. Workers hold private aggregate states that merge at the
   end of the pipeline. On CUDA with NVRTC, `fused::FusedPipelineRunner` compiles one kernel per
   pipeline shape from `fused::Codegen` and runs it over encoded pages.
7. Results are Arrow layout `RecordBatch`es in host memory, or moved to device when the caller asked
   to keep them for `interop::DlpackExporter`.

## Memory manager

`memory::MemoryManager` owns the tier hierarchy. Pages are column chunks (`PageDesc`), never fixed
size blocks. Bandwidth per tier pair is measured at startup by `ProbeBandwidth` and cached.

* Discrete mode: `LoadToHost` reads the chunk into pinned memory (posix) or `LoadToDevice` reads it
  straight into VRAM through cuFile; `PromoteHostToDevice` is an explicit async copy.
* Coherent mode: host memory is managed/ATS mapped; HBM residency is a `cudaMemAdvise` placement
  hint and eviction is a hint, not a copy.
* `EvictionPolicy` never touches pinned or scheduled pages and biases toward keeping small
  dimension tables resident.
* `SpillController` spills intermediates to pinned host memory first and NVMe second; hash join
  builds that exceed their budget radix partition and spill all but the resident partition.
* `PoolAllocator` uses size classes and a per query arena so a finished query returns memory as a unit.

## Storage

A segment (`storage::SegmentWriter`) is a horizontal slice with one contiguous encoded blob per
column, 4 KB aligned so a chunk is one `cuFileRead` and one decode launch. The footer carries the
encoding, checksum, zone map, distinct estimate, optional bloom filter and dictionary id per column.
`EncodingSelector` picks dictionary, FOR+bitpack, RLE or delta per column; general purpose codecs
wrap those only for cold tiers.

Predicates on dictionary or FOR columns are rewritten by `planner::PredicateRewriter` into the
encoded domain per chunk, with constant folding when the literal falls outside the chunk's domain.
RLE columns aggregate over runs with the run length as weight.

## Ingest and durability

`ingest::IngestService` writes to the `WriteAheadLog` (length prefixed, CRC32C, fsync) and then the
pinned `DeltaStore` before acknowledging. Queries union delta batches with base segments through
the scan source. The `Compactor` encodes delta batches into segments, swaps the catalog's segment
list atomically (old lists stay alive for in flight queries) and trims the WAL.
`durability::Recovery` reloads segment lists, rejects torn segment files, replays the WAL, and leaves
VRAM to refill on demand. `durability::FaultInjector` provides the kill points the chaos tests use.

## Multi GPU

`exchange::HashPartitioner` and `ExchangeOperator` shuffle or broadcast batches over a `Transport`
(in process for tests, NCCL on one node, UCX across nodes). `ResidencyDirectory` is the shared
residency map: the planner places a pipeline on the GPU that already holds most of its inputs and
registers partition ownership for copartitioned local joins.

## Operator sources

libcudf operators live behind `exec::OperatorBackend` (`CudfBackend`); the always working reference
is `CpuBackend`. Swapping is `EngineConfig::prefer_cudf_operators`. Fused kernels replace operator
chains only where `PipelineSplitter::IsFusable` says the shape is worth it.
