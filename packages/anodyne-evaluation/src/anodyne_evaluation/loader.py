"""Load a dataset-version artifact from object-store bytes into a DataFrame.

Tabular artifacts are Parquet (see the generation side's `register_version`);
text artifacts are JSONL. The evaluation activity fetches the bytes via the
`ObjectStore` port and hands them here so the judges receive plain DataFrames.
"""

from __future__ import annotations

import io

import pandas as pd  # type: ignore[import-untyped]


class UnsupportedArtifactError(ValueError):
    """Raised when an artifact's format can't be loaded for evaluation."""


def load_artifact(data: bytes, fmt: str) -> pd.DataFrame:
    """Parse `data` into a DataFrame, dispatching on the version `format`."""
    f = fmt.lower()
    if f == "parquet":
        return pd.read_parquet(io.BytesIO(data))
    if f in ("csv",):
        return pd.read_csv(io.BytesIO(data))
    if f in ("jsonl", "json"):
        return pd.read_json(io.BytesIO(data), lines=(f == "jsonl"))
    raise UnsupportedArtifactError(f"cannot load artifact of format {fmt!r} for evaluation")
