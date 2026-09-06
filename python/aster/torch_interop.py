"""Zero copy handoff of result columns to PyTorch and JAX through DLPack. Columns left on device
(keep_on_device=True) become CUDA tensors without a host round trip."""
from __future__ import annotations

from typing import Any, Optional


def _numeric_columns(result: Any) -> list[int]:
    return [i for i, t in enumerate(result.column_types) if t not in ("string", "binary")]


def to_torch(result: Any, column: Optional[int] = None) -> Any:
    import torch
    if column is not None:
        return torch.from_dlpack(result.dlpack(column))
    cols = _numeric_columns(result)
    if not cols:
        raise ValueError("no numeric columns to export")
    tensors = [torch.from_dlpack(result.dlpack(i)) for i in cols]
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack([t.to(torch.float64) if t.dtype != tensors[0].dtype else t for t in tensors], dim=1)


def to_jax(result: Any, column: Optional[int] = None) -> Any:
    import jax
    if column is not None:
        return jax.dlpack.from_dlpack(result.dlpack(column))
    cols = _numeric_columns(result)
    if not cols:
        raise ValueError("no numeric columns to export")
    arrays = [jax.dlpack.from_dlpack(result.dlpack(i)) for i in cols]
    if len(arrays) == 1:
        return arrays[0]
    import jax.numpy as jnp
    return jnp.stack(arrays, axis=1)


def to_cudf(result: Any) -> Any:
    import cudf
    return cudf.DataFrame.from_arrow(result.to_arrow())
