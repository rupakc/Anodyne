from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality
from anodyne_tabular.copula_generator import CopulaTabularGenerator
from anodyne_tabular.profiler import PandasSampleProfiler


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "score": rng.normal(50, 10, n),
            "is_active": rng.random(n) < 0.5,
            "plan": rng.choice(["gold", "silver"], n),
            "email": [f"user{i}@example.com" for i in range(n)],
        }
    )


def _spec(fields: list[FieldSpec], target_rows: int = 200) -> DatasetSpec:
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


def _build_generator(sample: pd.DataFrame) -> tuple[CopulaTabularGenerator, DatasetSpec]:
    profiler = PandasSampleProfiler()
    profile = profiler.profile_dataframe(uuid4(), uuid4(), "k", "s.csv", sample)
    generator = CopulaTabularGenerator(profile, sample)
    spec = _spec(generator._fields)  # noqa: SLF001 - reuse the profiler-derived schema
    return generator, spec


def test_generate_is_deterministic_for_same_seed_and_range() -> None:
    sample = _sample_df()
    generator, spec = _build_generator(sample)

    t1 = generator.generate(spec, 0, 50, seed=7)
    t2 = generator.generate(spec, 0, 50, seed=7)

    assert t1.equals(t2)
    assert t1.num_rows == 50
    assert set(t1.column_names) == {f.name for f in spec.fields}


def test_numeric_columns_stay_within_observed_bounds() -> None:
    sample = _sample_df()
    generator, spec = _build_generator(sample)

    table = generator.generate(spec, 0, 200, seed=3)

    age = table.column("age").to_pylist()
    assert all(sample["age"].min() <= v <= sample["age"].max() for v in age)


def test_email_column_is_faker_generated_not_copied_from_sample() -> None:
    sample = _sample_df()
    generator, spec = _build_generator(sample)

    table = generator.generate(spec, 0, 50, seed=9)

    emails = set(table.column("email").to_pylist())
    assert emails.isdisjoint(set(sample["email"]))


def test_disjoint_shards_produce_different_rows() -> None:
    sample = _sample_df()
    generator, spec = _build_generator(sample)

    a = generator.generate(spec, 0, 20, seed=5).column("age").to_pylist()
    b = generator.generate(spec, 20, 20, seed=5).column("age").to_pylist()

    assert a != b


def test_categorical_output_restricted_to_choices() -> None:
    sample = _sample_df()
    generator, spec = _build_generator(sample)

    table = generator.generate(spec, 0, 100, seed=1)

    assert set(table.column("plan").to_pylist()) <= {"gold", "silver"}
