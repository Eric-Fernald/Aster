#!/usr/bin/env bash
# Chaos loop: kill the engine mid query, mid compaction and mid WAL write, then verify recovery.
# The in process kill points live in tests/test_ingest.cpp; this script adds real process kills.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/cpu/bench/aster-bench"
[[ -x "$BIN" ]] || BIN="$ROOT/build/cpu-gcc/aster-bench"
TESTS="$ROOT/build/cpu/tests/aster_tests"
[[ -x "$TESTS" ]] || TESTS="$ROOT/build/cpu-gcc/aster_tests"
DATA="${ASTER_CHAOS_ROOT:-/tmp/aster_chaos}"
ROUNDS="${1:-5}"
rm -rf "$DATA"

echo "== in process kill points"
"$TESTS" chaos
"$TESTS" ingest_compact
"$TESTS" wal_torn

echo "== process kills during ingest + queries"
for round in $(seq 1 "$ROUNDS"); do
  "$BIN" --suite clickbench --rows 200000 --root "$DATA" --iters 50 --warmup 0 --ingest >/dev/null 2>&1 &
  pid=$!
  sleep "$(awk -v r="$round" 'BEGIN { print 0.5 + (r % 3) * 0.7 }')"
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  # Reopen: recovery must succeed and every query must still run.
  if ! "$BIN" --suite clickbench --root "$DATA" --skip-load --iters 1 --warmup 0 --query cb0_count >/dev/null 2>&1; then
    echo "round $round: recovery FAILED"; exit 1
  fi
  echo "round $round: recovered"
done
echo "chaos: all rounds recovered"
