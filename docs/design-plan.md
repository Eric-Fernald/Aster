# GPU Native Analytical Engine: Design Plan

**Working name:** Aster (placeholder)
**Status:** Proposed
**Date:** September 2026

---

## 0. Thesis

Every GPU database so far has been built around a copy step: pull data across PCIe, run kernels, ship results back. The GPU spends most of its life waiting on a bus that runs at roughly 1/50th of its own memory bandwidth. The hardware that removes this bottleneck now exists (cache coherent CPU to GPU links, direct NVMe to VRAM transfers, hardware decompression), but no engine has been designed for it from the ground up.

Aster is a GPU native SQL execution engine, not a database. It plugs into existing systems through Substrait and Arrow, treats GPU memory as the top tier of a coherent hierarchy rather than as a destination for copies, and executes on compressed data with fused kernels so that intermediates never touch VRAM.

Two bets, in order of importance:

1. **A memory manager designed for unified and tiered memory** beats a faster kernel library every time the working set exceeds VRAM.
2. **A drop in engine** gets adopted where a new database would not.

---

## 1. Requirements

### 1.1 Functional

- Accept Substrait plans from a host system (DuckDB first, Spark and Postgres later) and return Arrow record batches.
- Full coverage of TPC-H and ClickBench operator set: scan, filter, project, hash join (inner, left, semi, anti), hash aggregate, sort, limit, window functions (phase 2), string operations.
- Work on datasets larger than GPU memory with graceful degradation, not a cliff.
- Read Parquet directly, plus a native segment format optimized for GPU decode.
- Streaming append ingest without pausing queries.
- CPU fallback for any operator the GPU path does not support, so no query ever fails because of the engine.
- Zero copy handoff of result columns to ML frameworks via DLPack.

### 1.2 Nonfunctional

| Dimension | Target |
|---|---|
| Performance | 5x or better cost efficiency vs DuckDB on the same rental spend for TPC-H SF100 and ClickBench |
| Working set | 5x VRAM with no more than 3x slowdown vs fully resident |
| Ingest | 1 GB/s sustained append per node while serving queries |
| Latency | Subsecond for interactive filter and aggregate on 1B rows resident |
| Availability | Engine crash does not lose committed data; recovery under 60 seconds for 1 TB |
| Portability | CUDA first; hardware abstraction layer so an AMD or Intel backend is possible without a rewrite |

### 1.3 Constraints and assumptions

- Small team (2 to 4 engineers) for the first year. Reuse aggressively.
- Open source under Apache 2.0. Commercial differentiation, if any, comes later from managed hosting or enterprise features.
- Primary development hardware: one H100 or H200 class node with NVMe, plus periodic access to a Grace Hopper (GH200) node for unified memory work.
- Linux only.

### 1.4 Non goals

- **OLTP.** No row level locking, no point updates, no transactions beyond append and compaction. Branchy, latency sensitive workloads are the wrong shape for a GPU.
- **A SQL parser or optimizer.** The host system owns those. We consume physical plans.
- **A storage service.** We read files and object storage; we do not manage replication or a catalog.
- **Beating hand tuned CUDA on a single fully resident query.** Crystal already did that. The goal is real workloads, not benchmark theater.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Host system (DuckDB / Spark / Postgres)                     │
│  parser → optimizer → Substrait plan                         │
└─────────────────────────────┬────────────────────────────────┘
                              │ Substrait
┌─────────────────────────────▼────────────────────────────────┐
│  Integration layer                                           │
│  plan validation · capability check · CPU fallback routing   │
└─────────────────────────────┬────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────┐
│  GPU physical planner                                        │
│  bandwidth aware cost model · segment pruning · placement    │
│  pipeline breaking · fusion boundaries                       │
└─────────────────────────────┬────────────────────────────────┘
                              │ fused pipeline DAG
┌─────────────────────────────▼────────────────────────────────┐
│  Execution engine                                            │
│  tile scheduler · JIT fused kernels · libcudf operators      │
│  compressed execution · spill controller                     │
└──────────┬──────────────────────────────────┬────────────────┘
           │ page requests                    │ exchange
┌──────────▼───────────────────┐   ┌──────────▼────────────────┐
│  Memory manager              │   │  Multi GPU exchange       │
│  HBM ⇄ host ⇄ NVMe tiers     │   │  NCCL / NVLink aware      │
│  prefetch · eviction · GDS   │   │  partition routing        │
└──────────┬───────────────────┘   └───────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│  Storage                                                     │
│  native segments (encoded columns + zone maps + bloom)       │
│  Parquet reader · delta store · WAL · compaction             │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 Query data flow

1. Host system parses and optimizes SQL, emits a Substrait plan.
2. Integration layer checks every relation and function against the capability registry. Unsupported subtrees are marked for CPU execution and their outputs are treated as external Arrow inputs.
3. Planner consults segment metadata (zone maps, bloom filters, row counts) and prunes segments that cannot match. This happens before any data moves.
4. Planner splits the plan at pipeline breakers (hash build, sort, aggregate finalize) and assigns each pipeline to a fused kernel. It estimates bytes moved per tier and picks a placement: fully resident, streamed from host, or streamed from NVMe.
5. Memory manager issues prefetches for the first N tiles of each pipeline while the JIT compiles or fetches cached kernels.
6. Tile scheduler runs pipelines. Each tile stays in registers and shared memory from scan through the pipeline breaker. Compressed columns are decoded lazily and only for surviving rows.
7. Results materialize as Arrow batches in pinned host memory, or stay in VRAM if the caller requested a DLPack handle.

---

## 3. Component Deep Dives

### 3.1 Integration layer

Substrait is the contract. Arrow is the data format on both sides.

- Ship a DuckDB extension first. Sirius has proven this path works and the extension API is stable.
- Maintain a **capability registry**: a table of Substrait function URIs and relation types with a status of `gpu`, `cpu`, or `unsupported`. The registry is data, not code, so coverage gaps are visible and testable.
- Fallback is per subtree, not per query. A single unsupported string function should not push a 40 GB join back onto the CPU.
- Round trip cost of fallback (GPU → Arrow → CPU → Arrow → GPU) is charged in the cost model. Sometimes it is cheaper to run the whole pipeline on CPU. Let the numbers decide.

### 3.2 Storage format

Native segment format, with Parquet as an import path.

**Segment layout**

- A segment is a horizontal slice of a table, target 64 MB to 256 MB encoded, aligned to a multiple of the tile size.
- Each column chunk carries: encoding type, min/max, null count, distinct count estimate, optional bloom filter, and byte offsets into a single contiguous blob.
- The blob is laid out so a whole column chunk can be fetched with one `cuFileRead` and decoded by one kernel launch. No nested page structures that need CPU side parsing.

**Encodings, in priority order**

| Encoding | When | Operates without decode? |
|---|---|---|
| Dictionary | Low cardinality strings and ints | Yes: filter on codes, join on codes when dictionaries are shared |
| Frame of reference + bit packing | Ints, dates, timestamps | Yes: predicates rewritten into packed domain |
| Run length | Sorted or clustered columns | Yes: filter and aggregate on runs |
| Delta | Monotonic ints | Partial: prefix sum needed for random access |
| General purpose (LZ4, Zstd) | Everything else, as an outer layer | No: decode via nvCOMP or Blackwell decompression engine |

Rule: the lightweight encodings are the primary representation. General purpose compression wraps them only for cold tiers where the byte savings pay for the decode.

### 3.3 Memory manager

This is the core bet and gets the most engineering time.

**Tiers**

```
Tier 0  HBM            3 to 8 TB/s   80 to 192 GB
Tier 1  Host DRAM      PCIe: 64 GB/s | NVLink C2C: 900 GB/s   0.5 to 2 TB
Tier 2  NVMe (GDS)     10 to 50 GB/s aggregate   tens of TB
Tier 3  Object storage network bound   unbounded
```

**Design principles**

- **Pages are column chunks, not fixed size blocks.** The unit of movement is one encoded column chunk from one segment. This keeps transfers large and sequential.
- **Bandwidth is measured, not assumed.** At startup, benchmark every tier pair and store the results. The cost model reads these numbers. A PCIe Gen4 box and a GH200 should produce different plans automatically.
- **Two hardware modes, one interface.**
  - *Discrete mode* (PCIe attached GPU): explicit copies via `cudaMemcpyAsync` from pinned host buffers and `cuFileRead` for NVMe. Prefetch depth tuned to hide latency.
  - *Coherent mode* (Grace Hopper or equivalent): host memory is mapped into the GPU address space via HMM or ATS. Cold pages live in LPDDR5X and are pulled on demand at 900 GB/s. Eviction from HBM is a hint, not a copy. This is where the architecture pays off, and it is the mode to design for even while testing on discrete hardware.
- **Eviction policy is query aware.** The planner knows which pages the next pipeline needs. Pages already scheduled are pinned; everything else is subject to a frequency and recency policy with a bias toward keeping small dimension tables resident.
- **Spilling of intermediates** (hash tables, sort runs) goes to host pinned memory first, NVMe second. Spilled hash tables are partitioned so probes can proceed on the resident partition while others load.
- **Allocation** through RMM with a pool allocator. Fragmentation in VRAM is fatal; use size classes and arena per query.

**Page descriptor**

```cpp
struct PageDesc {
  uint64_t segment_id;
  uint32_t column_id;
  uint32_t encoded_bytes;
  uint8_t  tier;
  uint8_t  encoding;
  uint16_t pin_count;
  uint64_t last_access_epoch;
  void*    device_ptr;
  void*    host_ptr;
};
```

### 3.4 Execution engine

**Tile based, pipeline fused.**

- A tile is a fixed number of rows (start at 32K, tune) processed by one thread block from scan through the next pipeline breaker.
- Pipelines are compiled to a single kernel with NVRTC. Kernels are cached by a hash of the pipeline shape (operator sequence, types, encodings), not by query text, so structurally similar queries reuse compiled code.
- Within a tile, each operator is a device function invoked in sequence. Intermediate columns are shared memory arrays or registers. Nothing is written to global memory until the pipeline breaker.

**Compressed execution**

- Predicates on dictionary or frame of reference columns are rewritten by the planner into the encoded domain. `WHERE region = 'EU'` becomes `WHERE region_code = 3`.
- Late materialization: the pipeline carries a selection vector and only decodes payload columns for rows that survive all filters.
- Aggregation on run length columns aggregates over runs, multiplying by run length.

**Operator sources**

- Start with libcudf for joins, aggregates, sorts. It is mature and correct.
- Replace with custom fused kernels only where profiling shows libcudf's materialize between operators is the bottleneck. Expect this for filter → project → aggregate chains first, then hash probe.
- Keep both paths behind the same operator interface so swapping is a config flag.

**Hash join specifics**

- Build side always attempts full residency. If it does not fit, radix partition both sides on the GPU and process partition pairs.
- Probe side streams. It never needs to be resident.
- Shared dictionary join: if both columns use the same dictionary, join on codes and skip string comparison entirely.

### 3.5 Planner

The host optimizer already chose join order and pushed down predicates. Our planner handles what it cannot know: where the data is and what moving it costs.

Cost model inputs:

- Segment metadata (rows surviving pruning, encoded bytes per column).
- Measured tier bandwidths.
- Current residency map from the memory manager.
- Kernel cache hit or miss (JIT compile time is real, tens to hundreds of milliseconds).

Decisions:

- Which segments to skip entirely.
- Which pipelines fuse and where the breakers land.
- Placement per pipeline: resident, stream from host, stream from NVMe, or CPU fallback.
- Prefetch schedule.

The planner produces a physical plan with byte movement estimates attached. Log these. Comparing estimated vs actual bytes moved is the primary tuning loop.

### 3.6 Ingest

- Appends land in a **CPU side delta store**: Arrow batches in pinned host memory, WAL to NVMe before acknowledgement.
- Queries see delta store rows through a union with the base segments. The delta is small, so scanning it on CPU or shipping it to GPU per query is cheap.
- A **background compactor** encodes delta batches into native segments (encoding selection, zone maps, bloom filters), writes them to NVMe, and atomically swaps the segment list. Old segment lists are retained until in flight queries finish.
- Target: 1 GB/s append per node with compaction keeping the delta under 1 GB.

### 3.7 Multi GPU

Phase 5 work, but the design accounts for it from day one.

- Tables are hash partitioned across GPUs on a chosen key at load time. Copartitioned tables join locally with no exchange.
- Non copartitioned joins shuffle via NCCL. On NVSwitch nodes this runs at 900 GB/s or better per GPU and is often faster than PCIe to host. On PCIe only nodes, broadcast the smaller side instead.
- Cross node exchange over UCX with GPUDirect RDMA. Treat this as phase 6 and expect it to hurt.
- The memory manager becomes per GPU with a shared residency map so the planner can place a pipeline on the GPU that already holds its inputs.

### 3.8 ML interop

- Any result set can be returned as DLPack capsules pointing at device memory. PyTorch, JAX, and cuDF consume these with zero copy.
- Vector columns (fixed width float arrays) are a first class type. Phase 6 adds cuVS CAGRA index build and ANN search as operators so vector similarity joins run in the same engine.
- A Python client exposes `engine.query(sql).to_torch()` and friends. This is the demo that will get attention.

### 3.9 Durability

GPU memory is volatile and has ECC but no persistence. Nothing in VRAM is ever the only copy.

- WAL on NVMe for the delta store, fsynced before ack.
- Native segments are immutable once written. Corruption is detected by per chunk checksums verified on GPU during decode.
- Recovery: replay WAL into the delta store, reload segment list. VRAM is a cache and is rebuilt on demand.

---

## 4. Key Decisions

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Engine vs database | Engine behind Substrait | Standalone DB with own SQL | Adoption. A decade of standalone GPU databases got niche traction. A backend for DuckDB gets tried in an afternoon. |
| Operator library | libcudf first, custom kernels second | Custom everything | Correctness and time to first benchmark. Fusion gains come later and only where measured. |
| Memory model | Design for coherent (Grace Hopper), run on discrete | Design for PCIe only | Discrete hardware is what most people have today, but the coherent design is the differentiator and the direction the hardware is going. |
| Compression | Lightweight encodings primary, operate on encoded data | Decode to plain columns on load | Bandwidth is the bottleneck. Every decoded byte is a wasted transfer. |
| Vendor | CUDA first, HAL from day one | Vendor neutral (SYCL, Vulkan compute) | The libraries that make this feasible in a year are CUDA only. The HAL keeps the door open. |
| Pipeline model | Tile based with JIT fusion | Operator at a time with libcudf | Intermediates in VRAM are the enemy when VRAM is the scarce resource. |

---

## 5. Roadmap

Each phase has an exit criterion. Do not start the next phase until the criterion is met and measured.

### Phase 0: Spike (4 weeks)

- DuckDB extension that intercepts Substrait, runs a filter and aggregate via libcudf, returns Arrow.
- Benchmark harness: TPC-H SF10, ClickBench subset, bytes moved and kernel time instrumented from the start.
- Tier bandwidth microbenchmark tool.

**Exit:** TPC-H Q1 and Q6 run end to end on GPU. Harness produces a cost per query number.

### Phase 1: Single GPU, fully resident (3 months)

- Full TPC-H operator coverage via libcudf.
- Capability registry and per subtree CPU fallback.
- Native segment format writer and reader (uncompressed first).
- RMM pool allocator, arena per query.

**Exit:** All 22 TPC-H queries at SF100 resident in VRAM, 5x cost efficiency vs DuckDB on the same instance class.

### Phase 2: Tiered memory manager (3 months)

- Column chunk pages, residency map, prefetch, eviction.
- GPUDirect Storage path for NVMe.
- Spill controller for hash tables and sort runs.
- Coherent mode prototype on GH200.

**Exit:** TPC-H SF1000 (roughly 5x to 10x VRAM) completes with no more than 3x slowdown vs SF100 per row scanned. Degradation curve is published.

### Phase 3: Compressed execution and fusion (4 months)

- Dictionary, frame of reference, run length encodings in the segment format.
- Predicate rewrite into encoded domain.
- NVRTC pipeline fusion with kernel cache.
- Late materialization.

**Exit:** Bytes moved per TPC-H query drops by 3x or more vs phase 2. Fused pipelines beat libcudf operator chains on filter → aggregate by 2x.

### Phase 4: Ingest and pruning (2 months)

- Delta store, WAL, compactor.
- Zone maps and bloom filters written by compactor, consumed by planner.
- ClickBench full run (it is heavy on string filters and data skipping).

**Exit:** 1 GB/s sustained append while running ClickBench concurrently. ClickBench cost efficiency at 5x vs DuckDB.

### Phase 5: Multi GPU (3 months)

- Partitioned load, copartitioned local joins.
- NCCL exchange operator.
- Shared residency map and GPU aware placement.

**Exit:** TPC-H SF3000 on an 8 GPU node with near linear scaling on copartitioned joins.

### Phase 6: ML interop and hardening (ongoing)

- DLPack export, Python client.
- cuVS vector index and ANN operators.
- Cross node exchange via UCX.
- Chaos testing: kill the engine mid query, mid compaction, mid WAL write. Recover every time.

---

## 6. Benchmarks and Metrics

Track all of these from phase 0. Optimizing the wrong one is how projects die with great TPC-H numbers and no users.

| Metric | Why it matters |
|---|---|
| Cost per query (rental $ × wall time) | The only number a buyer cares about |
| Bytes moved per tier per query | The actual bottleneck; estimated vs actual is the tuning loop |
| HBM bandwidth utilization during scan | Below 60% means the GPU is waiting |
| Working set / VRAM ratio vs slowdown | The degradation curve is the product |
| Kernel cache hit rate | JIT compile time hides in tail latency |
| Ingest throughput under concurrent query load | Proves it is not a batch only toy |
| Fallback rate | Percentage of query subtrees hitting CPU; should trend to zero |

Benchmarks: TPC-H (SF100, SF1000, SF3000), ClickBench, and one real geospatial or time series dataset of at least 10B rows because synthetic data lies about compression ratios.

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sirius and Theseus already occupy the "GPU engine behind Substrait" space | High | High | Differentiate on the memory manager and coherent mode. If they solve tiering first, contribute upstream instead of competing. Decide by end of phase 2. |
| GH200 access is expensive and scarce | High | Medium | Design for coherent mode, validate on discrete, rent GH200 time in bursts for phase 2 and 3 exits |
| libcudf API churn | Medium | Medium | Pin versions, wrap behind the operator interface, budget one upgrade per phase |
| JIT complexity balloons | Medium | High | Fuse only pipeline shapes that profiling justifies. Keep the libcudf path as the always working fallback. |
| Substrait coverage gaps in DuckDB's producer | Medium | Medium | Capability registry makes gaps visible; contribute fixes upstream |
| GPU prices stay high because of AI demand | High | Medium | Cost per query metric already accounts for it. If the GPU is not 5x cheaper per query, the project should know early. |
| Team of 2 to 4 cannot deliver phase 5 and 6 | High | Low | Phases 0 to 4 are a complete, useful product on their own. Multi GPU is upside, not table stakes. |

---

## 8. What to Revisit as It Grows

- **Tile size and pipeline breaker placement** once real workloads replace TPC-H. Interactive dashboards want small tiles and low latency; batch ETL wants the opposite.
- **Operator library choice.** If custom fused kernels end up covering 80% of hot pipelines, libcudf becomes dead weight and the dependency should be dropped.
- **The HAL.** If an AMD MI300 class part becomes the price performance leader, the abstraction layer gets tested for real. Until then, do not spend on it beyond keeping the interface clean.
- **Catalog and metadata.** Segment metadata in local files works for one node. Multi node needs a real catalog, and that is the moment this stops being an engine and starts being a database whether anyone wanted that or not.
- **Coherent mode as default.** When most deployments run on coherent hardware, the discrete code path becomes the legacy branch. Plan the deprecation rather than carrying both forever.

---

## 9. Open Questions

1. Does HMM on Linux give good enough page fault performance for coherent mode on non Grace hardware, or is ATS on GH200 the only viable path?
2. How much of the encoded domain predicate rewrite can be done in the host optimizer (DuckDB) vs our planner? Doing it upstream benefits everyone but couples us to DuckDB internals.
3. Is Blackwell's hardware decompression engine accessible through nvCOMP or does it need direct driver work?
4. What is the right partitioning key selection strategy for multi GPU when the workload is unknown at load time? Sample queries first, or require the user to declare it?
5. Should the delta store be queryable on GPU from the start (ship pinned batches per query) or CPU only until compaction? Measure the crossover.
