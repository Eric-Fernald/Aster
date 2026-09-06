#!/usr/bin/env bash
# Direct g++ build of the CPU only configuration for machines without CMake. Mirrors the "cpu" preset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${BUILD_DIR:-$ROOT/build/cpu-gcc}"
CXX="${CXX:-g++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p "$OUT/obj"

obj_of() { echo "$OUT/obj/$(echo "${1#$ROOT/}" | tr '/' '_').o"; }

compile() {
  local src="$1"
  local obj
  obj="$(obj_of "$src")"
  if [[ ! -f "$obj" || "$src" -nt "$obj" ]] || find "$ROOT/include" "$ROOT/src" "$ROOT/tests" "$ROOT/bench" -newer "$obj" -name '*.hpp' | grep -q .; then
    echo "  CXX ${src#$ROOT/}"
    "$CXX" -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -fPIC -pthread \
      -I"$ROOT/include" -I"$ROOT/third_party" -I"$ROOT/bench" -I"$ROOT/tests" ${EXTRA_CXXFLAGS:-} \
      -c "$src" -o "$obj"
  fi
}
export -f compile obj_of
export OUT CXX ROOT EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

mapfile -t CORE < <(find "$ROOT/src" -name '*.cpp' ! -name 'c_api.cpp' | sort)
mapfile -t HARNESS < <(find "$ROOT/bench/harness" -name '*.cpp' | sort)
mapfile -t TESTS < <(find "$ROOT/tests" -name 'test_*.cpp' | sort)
TOOLS=("$ROOT/tools/aster_bwprobe.cpp" "$ROOT/tools/aster_segment_inspect.cpp" "$ROOT/tools/aster_capabilities.cpp")

printf '%s\0' "${CORE[@]}" "${HARNESS[@]}" "${TESTS[@]}" "$ROOT/src/c_api.cpp" "$ROOT/bench/aster_bench.cpp" "${TOOLS[@]}" \
  | xargs -0 -P "$JOBS" -I{} bash -c 'compile "$1"' _ {}

CORE_OBJS=(); for s in "${CORE[@]}"; do CORE_OBJS+=("$(obj_of "$s")"); done
HARNESS_OBJS=(); for s in "${HARNESS[@]}"; do HARNESS_OBJS+=("$(obj_of "$s")"); done
TEST_OBJS=(); for s in "${TESTS[@]}"; do TEST_OBJS+=("$(obj_of "$s")"); done

echo "  AR  libaster_core.a"
ar rcs "$OUT/libaster_core.a" "${CORE_OBJS[@]}"
echo "  LD  libaster.so"
"$CXX" -shared -o "$OUT/libaster.so" "$(obj_of "$ROOT/src/c_api.cpp")" "${CORE_OBJS[@]}" -pthread
echo "  LD  aster_tests"
"$CXX" -o "$OUT/aster_tests" "${TEST_OBJS[@]}" "$OUT/libaster_core.a" -pthread
echo "  LD  aster-bench"
"$CXX" -o "$OUT/aster-bench" "$(obj_of "$ROOT/bench/aster_bench.cpp")" "${HARNESS_OBJS[@]}" "$OUT/libaster_core.a" -pthread
for t in aster_bwprobe aster_segment_inspect aster_capabilities; do
  echo "  LD  ${t//_/-}"
  "$CXX" -o "$OUT/${t//_/-}" "$(obj_of "$ROOT/tools/$t.cpp")" "$OUT/libaster_core.a" -pthread
done
echo "build complete: $OUT"
