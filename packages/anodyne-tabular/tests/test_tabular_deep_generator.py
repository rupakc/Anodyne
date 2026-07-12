from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality
from anodyne_tabular.deep_generator import DeepTabularGenerator
from anodyne_tabular.profiler import PandasSampleProfiler


def _sample_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "plan": rng.choice(["gold", "silver"], n),
        }
    )


def _spec(fields: list[FieldSpec], target_rows: int = 120) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="sample",
        fields=fields,
        target_rows=target_rows,
    )


def test_unknown_kind_raises_at_construction() -> None:
    sample = _sample_df(10)
    profile = PandasSampleProfiler().profile_dataframe(uuid4(), uuid4(), "k", "s.csv", sample)

    with pytest.raises(ValueError):
        DeepTabularGenerator(profile, sample, kind="bogus")  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["ctgan", "tvae"])
def test_fit_and_sample_produces_expected_shape(kind: str) -> None:
    sample = _sample_df()
    profile = PandasSampleProfiler().profile_dataframe(uuid4(), uuid4(), "k", "s.csv", sample)
    generator = DeepTabularGenerator(profile, sample, kind=kind, epochs=1)  # type: ignore[arg-type]
    spec = _spec(generator._fields)  # noqa: SLF001

    table = generator.generate(spec, 0, 30, seed=1)

    assert table.num_rows == 30
    assert set(table.column_names) == {"age", "plan"}
    assert set(table.column("plan").to_pylist()) <= {"gold", "silver"}


@pytest.mark.integration
def test_generate_is_deterministic_for_same_seed_and_range() -> None:
    sample = _sample_df()
    profile = PandasSampleProfiler().profile_dataframe(uuid4(), uuid4(), "k", "s.csv", sample)
    generator = DeepTabularGenerator(profile, sample, kind="ctgan", epochs=1)
    spec = _spec(generator._fields)  # noqa: SLF001

    t1 = generator.generate(spec, 0, 20, seed=4)
    t2 = generator.generate(spec, 0, 20, seed=4)

    assert t1.equals(t2)
