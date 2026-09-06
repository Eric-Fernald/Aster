#!/usr/bin/env bash
# Builds (CMake if available, direct g++ otherwise) and runs the unit test binary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if command -v cmake >/dev/null 2>&1; then
  cmake --preset cpu >/dev/null
  cmake --build --preset cpu -j "$(nproc 2>/dev/null || echo 4)"
  exec "$ROOT/build/cpu/tests/aster_tests" "$@"
else
  bash "$ROOT/scripts/build_cpu.sh"
  exec "$ROOT/build/cpu-gcc/aster_tests" "$@"
fi
