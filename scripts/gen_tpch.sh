#!/usr/bin/env bash
# Produces TPC-H Parquet files with DuckDB's tpch extension for the host (Substrait) path, then
# imports them into an Aster data root so both engines see the same data.
# Usage: scripts/gen_tpch.sh <scale_factor> <out_dir>
set -euo pipefail
SF="${1:-1}"
OUT="${2:-/var/tmp/aster/tpch_sf$SF}"
mkdir -p "$OUT"
python3 - "$SF" "$OUT" <<'PY'
import sys, duckdb
sf, out = float(sys.argv[1]), sys.argv[2]
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute(f"CALL dbgen(sf={sf})")
for t in ["lineitem", "orders", "customer", "part", "supplier", "partsupp", "nation", "region"]:
    con.execute(f"COPY {t} TO '{out}/{t}.parquet' (FORMAT PARQUET)")
    print("wrote", t)
PY
echo "parquet files in $OUT"
echo "import with: python3 -c \"import aster; e=aster.Engine(root='$OUT/aster'); [e.import_parquet(t, '$OUT/'+t+'.parquet') for t in ['lineitem','orders','customer','part','supplier','partsupp','nation','region']]\""
