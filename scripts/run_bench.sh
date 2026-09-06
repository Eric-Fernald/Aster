#!/usr/bin/env bash
# Runs the TPC-H and ClickBench suites and writes results under bench/results/<timestamp>/.
# Usage: scripts/run_bench.sh [cpu|discrete|coherent] [sf] [clickbench_rows]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-cpu}"
SF="${2:-0.1}"
ROWS="${3:-1000000}"
BIN="$ROOT/build/cpu/bench/aster-bench"
[[ -x "$BIN" ]] || BIN="$ROOT/build/cpu-gcc/aster-bench"
[[ -x "$BIN" ]] || BIN="$ROOT/build/cuda/bench/aster-bench"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/bench/results/$STAMP"
mkdir -p "$OUT"
DATA="${ASTER_BENCH_ROOT:-/var/tmp/aster/bench}"

echo "== TPC-H SF$SF ($MODE)"
"$BIN" --suite tpch --mode "$MODE" --sf "$SF" --root "$DATA/tpch" --iters 3 --warmup 1 \
  --json "$OUT/tpch.json" --csv "$OUT/tpch.csv"
echo "== ClickBench $ROWS rows with concurrent ingest ($MODE)"
"$BIN" --suite clickbench --mode "$MODE" --rows "$ROWS" --root "$DATA/clickbench" --iters 3 --warmup 1 --ingest \
  --json "$OUT/clickbench.json" --csv "$OUT/clickbench.csv"

# Degradation curve: rerun TPC-H with shrinking VRAM budgets (working set / VRAM ratio vs slowdown).
if [[ "$MODE" != "cpu" ]]; then
  for MB in 65536 16384 8192 4096; do
    echo "== TPC-H SF$SF with ${MB} MB VRAM budget"
    "$BIN" --suite tpch --mode "$MODE" --sf "$SF" --root "$DATA/tpch" --skip-load --iters 2 --warmup 1 --vram-mb "$MB" \
      --json "$OUT/tpch_vram_${MB}.json"
  done
fi
echo "results in $OUT"
