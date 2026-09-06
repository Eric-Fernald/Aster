"""Aster: a GPU native SQL execution engine behind Substrait and Arrow.

    import aster
    engine = aster.Engine(mode="discrete", root="/var/tmp/aster")
    engine.import_parquet("lineitem", "lineitem.parquet")
    result = engine.query("SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1")
    result.to_torch()      # zero copy when the result stayed on device
"""
from .client import Engine, QueryResult, AsterError
from .duckdb_bridge import sql_to_substrait

__all__ = ["Engine", "QueryResult", "AsterError", "sql_to_substrait"]
__version__ = "0.1.0"
