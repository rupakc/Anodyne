"""Post-processing constraint enforcement applied after every tabular synthesizer.

Clips numeric columns to their declared/profiled min/max, restricts categorical columns to
their declared choice set (re-mapping out-of-vocabulary values to the most frequent declared
choice), and reorders/selects columns to exactly match `spec.fields`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import FieldSpec, SemanticType


def enforce(table: pa.Table, fields: list[FieldSpec]) -> pa.Table:
    """Return a new table with `fields`' constraints applied and columns in `fields` order."""
    columns: dict[str, pa.Array] = {}
    for field in fields:
        column = table.column(field.name)
        columns[field.name] = _enforce_column(column, field)
    return pa.table(columns)


def _enforce_column(column: pa.ChunkedArray, field: FieldSpec) -> pa.Array:
    c = field.constraints
    if field.semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT):
        if "min" not in c and "max" not in c:
            return column
        values = np.asarray(column.to_pylist(), dtype="float64")
        if "min" in c:
            values = np.clip(values, a_min=float(c["min"]), a_max=None)  # type: ignore[arg-type]
        if "max" in c:
            values = np.clip(values, a_min=None, a_max=float(c["max"]))  # type: ignore[arg-type]
        if field.semantic_type is SemanticType.INTEGER:
            return pa.array(values.astype("int64").tolist(), type=pa.int64())
        return pa.array(values.tolist(), type=pa.float64())
    if field.semantic_type is SemanticType.CATEGORICAL and "choices" in c:
        choices = set(cast("list[str]", c["choices"]))
        fallback = c.get("most_frequent") or (next(iter(choices)) if choices else None)
        mapped = [v if v in choices else fallback for v in column.to_pylist()]
        return pa.array(mapped)
    return column
