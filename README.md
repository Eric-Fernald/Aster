# Aster

A GPU native SQL execution engine, not a database. Hosts (DuckDB first) parse and optimize SQL and
hand Aster a Substrait plan; Aster returns Arrow batches, on host or left in VRAM for DLPack.

Two bets, in order: a memory manager built for unified and tiered memory beats a faster kernel
library once the working set exceeds VRAM, and a drop in engine gets adopted where a new database
would not. The full design is in [docs/design-plan.md](docs/design-plan.md).

## Layout

| Directory | What lives there |
|---|---|
| `include/aster/common` | Status/Result, types, Arrow layout `Column`/`RecordBatch`, config, hashing, CRC32C |
| `include/aster/hal` | Hardware abstraction: `Device` interface, CPU backend, CUDA backend |
| `include/aster/memory` | Page descriptors, residency map, bandwidth probe, query aware eviction, prefetcher, pool allocator, GDS/posix reader, spill controller, the memory manager (discrete and coherent modes) |
| `include/aster/storage` | Native segment format, encodings (plain, dictionary, FOR+bitpack, RLE, delta, general), zone maps + HLL, blocked bloom filters, encoding selector, Parquet import, catalog with atomic segment list swaps |
| `include/aster/integration` | Plan IR + `PlanBuilder`, capability registry (data, not code), Substrait consumer, per subtree CPU fallback router with round trip pricing, Arrow bridge |
| `include/aster/planner` | Bandwidth aware cost model, segment pruning, pipeline splitting, placement, encoded domain predicate rewrite, prefetch schedule |
| `include/aster/exec` | Tile model, expression evaluator, operator backends (CPU reference, libcudf), scan/join/aggregate/sort/window/vector operators, NVRTC codegen + kernel cache + JIT + fused runner, tile scheduler, executor |
| `include/aster/ingest` | WAL (CRC, fsync, torn tail recovery), pinned delta store, background compactor, ingest service |
| `include/aster/exchange` | Hash partitioner, in process / NCCL / UCX transports, exchange operator, shared residency directory |
| `include/aster/interop` | DLPack export/import |
| `include/aster/durability` | Recovery, fault injection points for chaos tests |
| `cuda/` | Hand written decode, filter/gather, hash join and partial aggregate kernels |
| `extensions/duckdb` | DuckDB loadable extension (`aster_query`, `aster_import_parquet`, `aster_capabilities`) |
| `python/` | `aster` package: ctypes client, DuckDB Substrait bridge, `to_torch()` / `to_jax()` |
| `bench/` | dbgen-lite TPC-H and ClickBench generators, IR queries, the harness, SQL for the host path |
| `tests/` | 64 unit and end to end tests |
| `tools/` | `aster-bwprobe`, `aster-segment-inspect`, `aster-capabilities` |

## Build

CPU only (no CUDA required; this is the reference and fallback path):

```bash
cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu
```

Without CMake:

```bash
scripts/build_cpu.sh && build/cpu-gcc/aster_tests
```

GPU (CUDA toolkit, libcudf/RMM, NVRTC, cuFile):

```bash
cmake --preset cuda && cmake --build --preset cuda
```

The `full` preset adds NCCL, UCX, nvCOMP, Substrait protobuf, DuckDB, LZ4/Zstd and cuVS. Every
optional dependency is an `ASTER_ENABLE_*` option; code paths that need them compile out cleanly.

## Run

```bash
build/cpu/aster-bench --suite tpch --sf 0.1 --iters 3 --json results.json
build/cpu/aster-bench --suite clickbench --rows 5000000 --ingest
build/cpu/aster-bwprobe --cuda --nvme /mnt/nvme/probe.bin --gds --out bandwidth.tsv
build/cpu/aster-capabilities --gaps
```

Python, with DuckDB providing the parser and optimizer:

```python
import aster
engine = aster.Engine(mode="discrete", root="/var/tmp/aster")
engine.import_parquet("lineitem", "lineitem.parquet")
aster.duckdb_bridge.register_parquet("lineitem", "lineitem.parquet")
r = engine.query("SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1", keep_on_device=True)
r.to_torch()
print(r.metrics)
```

## Metrics

Every query records: wall and kernel time, JIT compile time, bytes moved per tier pair (estimated
vs actual), rows scanned, segments pruned, kernel cache hit rate, fallback rate, spill bytes, HBM
bandwidth utilization and cost per query. See [docs/metrics.md](docs/metrics.md).

## Roadmap

Phases 0 through 6 with exit criteria are tracked in [docs/roadmap.md](docs/roadmap.md).

Apache 2.0.
