from __future__ import annotations

import io

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_tabular.io import UnsupportedSampleFormatError, read_sample


def _csv_bytes() -> bytes:
    return b"age,name\n30,Alice\n40,Bob\n"


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def test_read_sample_csv() -> None:
    df = read_sample(_csv_bytes(), "sample.csv")
    assert list(df.columns) == ["age", "name"]
    assert df["age"].tolist() == [30, 40]
    assert df["name"].tolist() == ["Alice", "Bob"]


def test_read_sample_parquet() -> None:
    original = pd.DataFrame({"age": [30, 40], "name": ["Alice", "Bob"]})
    df = read_sample(_parquet_bytes(original), "sample.parquet")
    assert list(df.columns) == ["age", "name"]
    assert df["age"].tolist() == [30, 40]


def test_read_sample_unknown_extension_raises() -> None:
    with pytest.raises(UnsupportedSampleFormatError):
        read_sample(b"whatever", "sample.xlsx")
