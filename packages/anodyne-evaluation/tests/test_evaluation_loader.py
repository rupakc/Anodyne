from __future__ import annotations

import io

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.loader import UnsupportedArtifactError, load_artifact


def test_loads_parquet() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    out = load_artifact(buf.getvalue(), "parquet")
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 3


def test_loads_jsonl() -> None:
    data = b'{"a": 1}\n{"a": 2}\n'
    out = load_artifact(data, "jsonl")
    assert out["a"].tolist() == [1, 2]


def test_unsupported_format_raises() -> None:
    with pytest.raises(UnsupportedArtifactError):
        load_artifact(b"...", "mp4")
