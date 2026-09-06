# Metrics

Tracked from phase 0 on every query. `metrics::QueryMetrics` carries them; `aster-bench` writes
them to JSON/CSV; `Engine::history()` returns the in process log.

| Metric | Field | Source |
|---|---|---|
| Cost per query (rental $ x wall time) | `cost_usd` | `wall_ms` x `EngineConfig::gpu_hourly_cost_usd` (or the CPU rate in cpu mode) |
| Bytes moved per tier per query | `bytes_nvme_to_host`, `bytes_nvme_to_hbm`, `bytes_host_to_hbm`, `bytes_hbm_to_host` | `MemoryManager` stats delta around the query |
| Estimated vs actual bytes | `bytes_estimated` vs `bytes_moved()` | planner estimate from residency + zone maps; the primary tuning loop |
| HBM bandwidth utilization during scan | `hbm_bandwidth_utilization` | bytes scanned / kernel time / measured HBM bandwidth; below 0.6 means the GPU is waiting |
| Working set / VRAM ratio vs slowdown | derived | run the same suite with `--vram-mb` sweeps; the degradation curve is the product |
| Kernel cache hit rate | `kernel_cache_hits`, `kernel_cache_misses` | `KernelCache` probe at plan time per fusable pipeline |
| JIT compile time | `jit_compile_ms` | NVRTC compile time hides in tail latency |
| Ingest throughput under concurrent query load | `aster-bench --ingest` | `RunIngestLoad` alongside the suite |
| Fallback rate | `subtrees_cpu / subtrees_total` | placement per relation after routing; should trend to zero |
| Segments pruned | `segments_pruned`, `segments_scanned` | zone map and bloom pruning before any data moves |
| Spill bytes | `spill_bytes` | `SpillController::total_spilled()` |

Process wide counters (`metrics::Registry::Global().Snapshot()`): `bytes_nvme_to_host`,
`bytes_host_to_hbm`, `bytes_nvme_to_hbm`, `evictions`, `rows_scanned`, `bytes_scanned_hbm`,
`spill_bytes`.

Benchmarks: TPC-H (SF100, SF1000, SF3000), ClickBench, and one real geospatial or time series
dataset of at least 10B rows, because synthetic data lies about compression ratios. The harness
generators exist so the pipeline is exercised end to end on any machine; the exit criteria are
measured on real data.
