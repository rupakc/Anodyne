from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from anodyne_dataset.models import SemanticType
from anodyne_tabular.profiler import PandasSampleProfiler


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    age = rng.integers(18, 80, n)
    score = rng.normal(50, 10, n)
    score = score.astype(object)
    score[0] = None  # introduce a null
    is_active = rng.random(n) < 0.5
    plan = rng.choice(["gold", "silver"], n)
    email = [f"user{i}@example.com" for i in range(n)]
    signup_at = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "age": age,
            "score": pd.array(score, dtype="float64"),
            "is_active": is_active,
            "plan": plan,
            "email": email,
            "signup_at": signup_at,
        }
    )


def test_profiler_from_dataframe_infers_types_and_stats() -> None:
    df = _sample_df()
    profiler = PandasSampleProfiler()
    tenant_id, dataset_id = uuid4(), uuid4()

    profile = profiler.profile_dataframe(tenant_id, dataset_id, "k", "s.csv", df)

    by_name = {c.name: c for c in profile.columns}
    age = by_name["age"]
    assert age.semantic_type is SemanticType.INTEGER
    assert age.min == float(df["age"].min())
    assert age.max == float(df["age"].max())
    assert age.mean is not None and age.std is not None

    score = by_name["score"]
    assert score.semantic_type is SemanticType.FLOAT
    assert score.null_rate > 0.0

    assert by_name["is_active"].semantic_type is SemanticType.BOOLEAN

    plan = by_name["plan"]
    assert plan.semantic_type is SemanticType.CATEGORICAL
    assert plan.categories is not None
    assert set(plan.categories) <= {"gold", "silver"}
    assert abs(sum(plan.categories.values()) - 1.0) < 1e-6

    assert by_name["email"].semantic_type is SemanticType.EMAIL
    assert by_name["signup_at"].semantic_type is SemanticType.DATETIME

    assert profile.row_count == len(df)
    assert profile.sample_uri == "k"
    assert profile.sample_filename == "s.csv"
    assert profile.tenant_id == tenant_id
    assert profile.dataset_id == dataset_id


def test_profiler_correlations_are_symmetric() -> None:
    df = _sample_df()
    profile = PandasSampleProfiler().profile_dataframe(uuid4(), uuid4(), "k", "s.csv", df)

    assert profile.correlations["age"]["score"] == profile.correlations["score"]["age"]


def test_profiler_caps_large_samples() -> None:
    big = _sample_df(500)
    profiler = PandasSampleProfiler(max_profile_rows=100)

    profile = profiler.profile_dataframe(uuid4(), uuid4(), "k", "s.csv", big)

    assert profile.row_count == 100


def test_profile_bytes_entrypoint_parses_csv() -> None:
    csv_bytes = b"age,name\n30,Alice\n40,Bob\n"
    profiler = PandasSampleProfiler()

    profile = profiler.profile(uuid4(), uuid4(), "k", csv_bytes, "s.csv")

    by_name = {c.name: c for c in profile.columns}
    assert by_name["age"].semantic_type is SemanticType.INTEGER
    assert profile.row_count == 2
