"""Load a dataset-version artifact from object-store bytes into a DataFrame.

Tabular artifacts are Parquet (see the generation side's `register_version`);
text artifacts are JSONL. The evaluation activity fetches the bytes via the
`ObjectStore` port and hands them here so the judges receive plain DataFrames.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from anodyne_graph.models import GraphDataset


class UnsupportedArtifactError(ValueError):
    """Raised when an artifact's format can't be loaded for evaluation."""


def load_graph(data: bytes) -> GraphDataset:
    """Parse a ``graph_json`` node-link artifact into a `GraphDataset`.

    Graph versions are not columnar, so they bypass the DataFrame `load_artifact`
    path; the graph judges receive the `GraphDataset` directly via the evaluation
    context. Imported lazily to keep this module's base import cheap.
    """
    from anodyne_graph.serialization import from_json_bytes

    return from_json_bytes(data)


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


def load_manifest(data: bytes) -> pd.DataFrame:
    """Parse a media dataset's manifest JSON into a records DataFrame.

    Accepts either ``{"items": [...]}`` or a bare ``[...]`` list.
    """
    doc = json.loads(data.decode("utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    return pd.DataFrame.from_records(items or [])
