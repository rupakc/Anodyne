import io
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import ray
from anodyne_compute.ray_tasks import generate_shard_bytes, remote_generate_shard
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType

pytestmark = pytest.mark.integration


def _spec() -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=20,
    )


def test_generate_shard_bytes_is_parquet() -> None:
    data = generate_shard_bytes(_spec(), 0, 20, 3)
    tbl = pq.read_table(io.BytesIO(data))
    assert tbl.num_rows == 20


def test_ray_remote_matches_local() -> None:
    ray.init(ignore_reinit_error=True)
    try:
        local = generate_shard_bytes(_spec(), 0, 20, 3)
        remote = ray.get(remote_generate_shard.remote(_spec(), 0, 20, 3))
        assert local == remote
    finally:
        ray.shutdown()
