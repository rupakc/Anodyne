from __future__ import annotations

import io
from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import ray
from anodyne_compute.sample_tasks import (
    generate_shard_bytes_from_generator,
    remote_generate_shard_from_generator,
)
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_tabular.copula_generator import CopulaTabularGenerator
from anodyne_tabular.profiler import PandasSampleProfiler

pytestmark = pytest.mark.integration


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"age": rng.integers(18, 80, 50)})


def _spec_and_generator() -> tuple[DatasetSpec, CopulaTabularGenerator]:
    sample = _sample()
    profile = PandasSampleProfiler().profile_dataframe(uuid4(), uuid4(), "k", "s.csv", sample)
    generator = CopulaTabularGenerator(profile, sample)
    spec = DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="sample",
        fields=generator._fields,  # noqa: SLF001
        target_rows=20,
    )
    return spec, generator


def test_generate_shard_bytes_from_generator_is_parquet() -> None:
    spec, generator = _spec_and_generator()

    data = generate_shard_bytes_from_generator(generator, spec, 0, 20, seed=3)

    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 20


def test_ray_remote_matches_local() -> None:
    spec, generator = _spec_and_generator()
    ray.init(ignore_reinit_error=True)
    try:
        local = generate_shard_bytes_from_generator(generator, spec, 0, 20, seed=3)
        remote = ray.get(remote_generate_shard_from_generator.remote(generator, spec, 0, 20, 3))
        assert local == remote
    finally:
        ray.shutdown()
