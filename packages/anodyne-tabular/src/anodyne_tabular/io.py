"""Read an uploaded tabular sample (CSV/Parquet bytes) into a `pandas.DataFrame`."""

from __future__ import annotations

import io

import pandas as pd  # type: ignore[import-untyped]


class UnsupportedSampleFormatError(ValueError):
    """Raised when a sample's filename extension isn't a supported tabular format."""


def read_sample(data: bytes, filename: str) -> pd.DataFrame:
    """Parse `data` as CSV or Parquet, dispatching on `filename`'s extension.

    Args:
        data: Raw file bytes.
        filename: Original filename (only the extension is used).

    Returns:
        The parsed sample as a `pandas.DataFrame`.

    Raises:
        UnsupportedSampleFormatError: If the extension isn't `.csv` or `.parquet`.
    """
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if lower.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    raise UnsupportedSampleFormatError(
        f"unsupported sample format for {filename!r}; expected .csv or .parquet"
    )
