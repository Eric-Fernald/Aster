"""ctypes binding over libaster's C API. The host (DuckDB) turns SQL into Substrait; this client
ships the plan to the engine and exposes results as Arrow, NumPy, or DLPack capsules."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from typing import Any, Optional

from .duckdb_bridge import sql_to_substrait


class AsterError(RuntimeError):
    pass


def _find_library() -> str:
    env = os.environ.get("ASTER_LIBRARY")
    if env:
        return env
    names = ["libaster.so", "libaster.dylib", "aster.dll"]
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.join(here, "..", ".."), os.path.join(here, "..", "..", "build", "cpu"),
             os.path.join(here, "..", "..", "build", "cpu-gcc"), os.path.join(here, "..", "..", "build", "cuda"), "/usr/local/lib", "/usr/lib"]
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p):
                return p
    raise AsterError("libaster not found; set ASTER_LIBRARY to the shared library path")


class _Lib:
    _instance: Optional["_Lib"] = None

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(_find_library())
        L = self.lib
        cp = ctypes.c_char_p
        vp = ctypes.c_void_p
        L.aster_version.restype = cp
        L.aster_engine_open.restype = vp
        L.aster_engine_open.argtypes = [ctypes.c_int, cp, ctypes.c_int, ctypes.POINTER(cp)]
        L.aster_engine_close.argtypes = [vp]
        L.aster_free_string.argtypes = [cp]
        L.aster_create_table_json.restype = ctypes.c_int
        L.aster_create_table_json.argtypes = [vp, cp, cp, ctypes.POINTER(cp)]
        L.aster_import_parquet.restype = ctypes.c_int
        L.aster_import_parquet.argtypes = [vp, cp, cp, ctypes.POINTER(cp)]
        L.aster_compact.restype = ctypes.c_int
        L.aster_compact.argtypes = [vp, cp, ctypes.POINTER(cp)]
        L.aster_query_substrait.restype = vp
        L.aster_query_substrait.argtypes = [vp, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(cp)]
        L.aster_query_substrait_json.restype = vp
        L.aster_query_substrait_json.argtypes = [vp, cp, ctypes.c_int, ctypes.POINTER(cp)]
        L.aster_result_free.argtypes = [vp]
        L.aster_result_num_rows.restype = ctypes.c_int64
        L.aster_result_num_rows.argtypes = [vp]
        L.aster_result_num_columns.restype = ctypes.c_int
        L.aster_result_num_columns.argtypes = [vp]
        L.aster_result_column_name.restype = cp
        L.aster_result_column_name.argtypes = [vp, ctypes.c_int]
        L.aster_result_column_type.restype = cp
        L.aster_result_column_type.argtypes = [vp, ctypes.c_int]
        L.aster_result_metrics_json.restype = vp
        L.aster_result_metrics_json.argtypes = [vp]
        L.aster_result_column_dlpack.restype = vp
        L.aster_result_column_dlpack.argtypes = [vp, ctypes.c_int, ctypes.POINTER(cp)]
        L.aster_result_column_arrow.restype = ctypes.c_int
        L.aster_result_column_arrow.argtypes = [vp, ctypes.c_int, vp, vp, ctypes.POINTER(cp)]
        L.aster_result_to_tsv.restype = vp
        L.aster_result_to_tsv.argtypes = [vp, ctypes.c_int64]
        L.aster_capabilities_tsv.restype = vp
        L.aster_capabilities_tsv.argtypes = [vp]
        L.aster_bandwidth_table.restype = vp
        L.aster_bandwidth_table.argtypes = [vp]

    @classmethod
    def get(cls) -> "_Lib":
        if cls._instance is None:
            cls._instance = _Lib()
        return cls._instance

    def take_string(self, ptr: int) -> str:
        if not ptr:
            return ""
        s = ctypes.string_at(ptr).decode()
        self.lib.aster_free_string(ctypes.c_char_p(ptr))
        return s


def _check(err: "ctypes.c_char_p", rc: Any) -> None:
    if err.value is not None:
        msg = err.value.decode()
        _Lib.get().lib.aster_free_string(err)
        raise AsterError(msg)
    if rc is None or (isinstance(rc, int) and rc < 0):
        raise AsterError("aster call failed")


_MODES = {"cpu": 0, "discrete": 1, "coherent": 2}


class QueryResult:
    def __init__(self, handle: int, keep_on_device: bool) -> None:
        self._h = handle
        self._lib = _Lib.get()
        self.on_device = keep_on_device

    def __del__(self) -> None:
        if getattr(self, "_h", None):
            self._lib.lib.aster_result_free(self._h)
            self._h = 0

    @property
    def num_rows(self) -> int:
        return self._lib.lib.aster_result_num_rows(self._h)

    @property
    def column_names(self) -> list[str]:
        n = self._lib.lib.aster_result_num_columns(self._h)
        return [self._lib.lib.aster_result_column_name(self._h, i).decode() for i in range(n)]

    @property
    def column_types(self) -> list[str]:
        n = self._lib.lib.aster_result_num_columns(self._h)
        return [self._lib.lib.aster_result_column_type(self._h, i).decode() for i in range(n)]

    @property
    def metrics(self) -> dict:
        return json.loads(self._lib.take_string(self._lib.lib.aster_result_metrics_json(self._h)))

    def to_tsv(self, max_rows: int = -1) -> str:
        return self._lib.take_string(self._lib.lib.aster_result_to_tsv(self._h, max_rows))

    def dlpack(self, column: int) -> Any:
        """A DLPack capsule for one numeric column; consumers take ownership."""
        err = ctypes.c_char_p()
        ptr = self._lib.lib.aster_result_column_dlpack(self._h, column, ctypes.byref(err))
        _check(err, ptr)
        pythonapi = ctypes.pythonapi
        pythonapi.PyCapsule_New.restype = ctypes.py_object
        pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return pythonapi.PyCapsule_New(ptr, b"dltensor", None)

    def to_torch(self, column: Optional[int] = None) -> Any:
        from .torch_interop import to_torch
        return to_torch(self, column)

    def to_jax(self, column: Optional[int] = None) -> Any:
        from .torch_interop import to_jax
        return to_jax(self, column)

    def to_arrow(self) -> Any:
        import pyarrow as pa
        from pyarrow.cffi import ffi
        arrays = []
        for i, name in enumerate(self.column_names):
            c_array = ffi.new("struct ArrowArray*")
            c_schema = ffi.new("struct ArrowSchema*")
            err = ctypes.c_char_p()
            rc = self._lib.lib.aster_result_column_arrow(self._h, i, int(ffi.cast("uintptr_t", c_array)), int(ffi.cast("uintptr_t", c_schema)), ctypes.byref(err))
            _check(err, rc)
            arrays.append(pa.Array._import_from_c(int(ffi.cast("uintptr_t", c_array)), int(ffi.cast("uintptr_t", c_schema))))
        return pa.table(arrays, names=self.column_names)

    def to_pandas(self) -> Any:
        return self.to_arrow().to_pandas()

    def __repr__(self) -> str:
        return f"QueryResult(rows={self.num_rows}, columns={self.column_names})"


class Engine:
    def __init__(self, mode: str = "discrete", root: Optional[str] = None, device_id: int = 0, duckdb_connection: Any = None) -> None:
        self._lib = _Lib.get()
        err = ctypes.c_char_p()
        self._h = self._lib.lib.aster_engine_open(_MODES[mode], (root or "").encode(), device_id, ctypes.byref(err))
        _check(err, self._h)
        self.mode = mode
        self._duck = duckdb_connection

    def close(self) -> None:
        if self._h:
            self._lib.lib.aster_engine_close(self._h)
            self._h = 0

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def version(self) -> str:
        return self._lib.lib.aster_version().decode()

    def create_table(self, name: str, schema: dict[str, str]) -> None:
        err = ctypes.c_char_p()
        payload = json.dumps([{"name": k, "type": v} for k, v in schema.items()])
        rc = self._lib.lib.aster_create_table_json(self._h, name.encode(), payload.encode(), ctypes.byref(err))
        _check(err, rc)

    def import_parquet(self, table: str, path: str) -> None:
        err = ctypes.c_char_p()
        rc = self._lib.lib.aster_import_parquet(self._h, table.encode(), path.encode(), ctypes.byref(err))
        _check(err, rc)

    def compact(self, table: str) -> None:
        err = ctypes.c_char_p()
        rc = self._lib.lib.aster_compact(self._h, table.encode(), ctypes.byref(err))
        _check(err, rc)

    def query_substrait(self, plan: bytes, keep_on_device: bool = False) -> QueryResult:
        err = ctypes.c_char_p()
        h = self._lib.lib.aster_query_substrait(self._h, plan, len(plan), int(keep_on_device), ctypes.byref(err))
        _check(err, h)
        return QueryResult(h, keep_on_device)

    def query_substrait_json(self, plan_json: str, keep_on_device: bool = False) -> QueryResult:
        err = ctypes.c_char_p()
        h = self._lib.lib.aster_query_substrait_json(self._h, plan_json.encode(), int(keep_on_device), ctypes.byref(err))
        _check(err, h)
        return QueryResult(h, keep_on_device)

    def query(self, sql: str, keep_on_device: bool = False) -> QueryResult:
        """SQL goes through DuckDB's parser and optimizer; Aster executes the Substrait plan."""
        plan = sql_to_substrait(sql, self._duck)
        return self.query_substrait(plan, keep_on_device)

    def capabilities(self) -> list[tuple[str, str, str]]:
        text = self._lib.take_string(self._lib.lib.aster_capabilities_tsv(self._h))
        rows = []
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
        return rows

    def bandwidth_table(self) -> str:
        return self._lib.take_string(self._lib.lib.aster_bandwidth_table(self._h))
