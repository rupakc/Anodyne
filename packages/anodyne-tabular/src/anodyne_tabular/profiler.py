"""`SampleProfiler` adapter: infers a `Profile` from an uploaded CSV/Parquet sample.

Heuristics only look at column name + pandas dtype -- no LLM involved, so this runs
synchronously and cheaply even on large uploads (capped by `max_profile_rows`).
"""

from __future__ import annotations

import re
from uuid import UUID, uuid4

import pandas as pd  # type: ignore[import-untyped]
from anodyne_dataset.models import ColumnProfile, Profile, SemanticType
from anodyne_dataset.ports import SampleProfiler

from anodyne_tabular.io import read_sample

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class PandasSampleProfiler(SampleProfiler):
    """Pandas-based `SampleProfiler`.

    Args:
        max_categories: object columns with at most this many distinct values are
            treated as categorical; above it, they're treated as free text.
        max_profile_rows: samples larger than this are randomly subsampled (seeded,
            for determinism) before profiling, bounding memory/CPU on huge uploads.
        max_category_entries: cap on how many category frequencies are recorded.
    """

    def __init__(
        self,
        *,
        max_categories: int = 50,
        max_profile_rows: int = 200_000,
        max_category_entries: int = 50,
    ) -> None:
        self._max_categories = max_categories
        self._max_profile_rows = max_profile_rows
        self._max_category_entries = max_category_entries

    def profile(
        self, tenant_id: UUID, dataset_id: UUID, sample_uri: str, data: bytes, filename: str
    ) -> Profile:
        df = read_sample(data, filename)
        return self.profile_dataframe(tenant_id, dataset_id, sample_uri, filename, df)

    def profile_dataframe(
        self,
        tenant_id: UUID,
        dataset_id: UUID,
        sample_uri: str,
        filename: str,
        df: pd.DataFrame,
    ) -> Profile:
        if len(df) > self._max_profile_rows:
            df = df.sample(n=self._max_profile_rows, random_state=0).reset_index(drop=True)

        columns = [self._column_profile(name, df[name]) for name in df.columns]
        numeric_names = [
            c.name for c in columns if c.semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT)
        ]
        correlations: dict[str, dict[str, float]] = {}
        if len(numeric_names) > 1:
            corr = df[numeric_names].apply(pd.to_numeric, errors="coerce").corr()
            correlations = {
                row: {col: float(corr.loc[row, col]) for col in corr.columns} for row in corr.index
            }

        return Profile(
            id=uuid4(),
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            row_count=len(df),
            columns=columns,
            correlations=correlations,
            sample_uri=sample_uri,
            sample_filename=filename,
        )

    def _column_profile(self, name: str, series: pd.Series) -> ColumnProfile:
        null_rate = float(series.isna().mean()) if len(series) else 0.0
        non_null = series.dropna()
        semantic_type = self._infer_semantic_type(name, series, non_null)

        kwargs: dict[str, object] = {
            "name": name,
            "semantic_type": semantic_type,
            "nullable": null_rate > 0.0,
            "null_rate": null_rate,
            "distinct_count": int(non_null.nunique()),
        }
        if semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric):
                kwargs["min"] = float(numeric.min())
                kwargs["max"] = float(numeric.max())
                kwargs["mean"] = float(numeric.mean())
                kwargs["std"] = float(numeric.std()) if len(numeric) > 1 else 0.0
        elif semantic_type is SemanticType.CATEGORICAL:
            counts = non_null.astype(str).value_counts(normalize=True)
            kwargs["categories"] = {
                str(k): float(v) for k, v in counts.head(self._max_category_entries).items()
            }
        return ColumnProfile(**kwargs)  # type: ignore[arg-type]

    def _infer_semantic_type(
        self, name: str, series: pd.Series, non_null: pd.Series
    ) -> SemanticType:
        lower = name.lower()
        if pd.api.types.is_bool_dtype(series):
            return SemanticType.BOOLEAN
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME
        if pd.api.types.is_integer_dtype(series):
            return SemanticType.INTEGER
        if pd.api.types.is_float_dtype(series):
            return SemanticType.FLOAT
        if not len(non_null):
            return SemanticType.TEXT
        sample = non_null.astype(str)
        if "email" in lower and sample.map(lambda v: bool(_EMAIL_RE.match(v))).mean() > 0.9:
            return SemanticType.EMAIL
        if "email" in lower:
            return SemanticType.EMAIL
        if "address" in lower:
            return SemanticType.ADDRESS
        if "name" in lower:
            return SemanticType.NAME
        if sample.nunique() <= self._max_categories:
            return SemanticType.CATEGORICAL
        return SemanticType.TEXT
