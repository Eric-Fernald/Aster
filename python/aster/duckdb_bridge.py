"""SQL to Substrait through DuckDB. The host system owns parsing and optimization; Aster consumes
the physical plan. Tables must exist in the DuckDB catalog (a view over Parquet is enough) so
DuckDB can resolve names and types; Aster resolves the same names against its own catalog."""
from __future__ import annotations

from typing import Any, Optional

_connection: Any = None


def default_connection() -> Any:
    global _connection
    if _connection is None:
        import duckdb
        _connection = duckdb.connect()
        try:
            _connection.execute("INSTALL substrait; LOAD substrait;")
        except Exception as e:  # pragma: no cover - depends on network / platform
            raise RuntimeError("DuckDB substrait extension unavailable: " + str(e)) from e
    return _connection


def register_parquet(table: str, path: str, connection: Optional[Any] = None) -> None:
    con = connection or default_connection()
    con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')")


def register_schema(table: str, schema: dict[str, str], connection: Optional[Any] = None) -> None:
    """Empty table with Aster's types so DuckDB can plan against it without any data."""
    con = connection or default_connection()
    mapping = {"bool": "BOOLEAN", "i8": "TINYINT", "i16": "SMALLINT", "i32": "INTEGER", "i64": "BIGINT", "u8": "UTINYINT", "u16": "USMALLINT",
               "u32": "UINTEGER", "u64": "UBIGINT", "f32": "FLOAT", "f64": "DOUBLE", "date32": "DATE", "timestamp": "TIMESTAMP", "string": "VARCHAR", "binary": "BLOB"}
    cols = ", ".join(f"{k} {mapping.get(v, 'VARCHAR')}" for k, v in schema.items())
    con.execute(f"CREATE OR REPLACE TABLE {table} ({cols})")


def sql_to_substrait(sql: str, connection: Optional[Any] = None) -> bytes:
    con = connection or default_connection()
    row = con.execute("CALL get_substrait(?)", [sql]).fetchone()
    if row is None:
        raise RuntimeError("get_substrait returned no plan")
    return bytes(row[0])


def sql_to_substrait_json(sql: str, connection: Optional[Any] = None) -> str:
    con = connection or default_connection()
    row = con.execute("CALL get_substrait_json(?)", [sql]).fetchone()
    if row is None:
        raise RuntimeError("get_substrait_json returned no plan")
    return row[0]
