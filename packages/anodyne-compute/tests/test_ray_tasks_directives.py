"""Confirms `generate_shard_bytes` applies `GenerationDirective`s (C6) via `DirectiveGenerator`,
and stays unchanged for directive-free specs (regression guard on C0's behavior).

Deliberately not marked `integration`: `generate_shard_bytes` is plain Python/pyarrow and needs
no live Ray cluster (only calling the `@ray.remote`-wrapped `remote_generate_shard.remote(...)`
does -- see the existing `test_ray_tasks.py`), so this stays in the fast unit lane.
"""

from __future__ import annotations

import io
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_compute.ray_tasks import generate_shard_bytes
from anodyne_dataset.directives import DirectiveKind, GenerationDirective, dump_directives
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType


def _spec(directives: dict[str, object] | None = None) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="flag", semantic_type=SemanticType.BOOLEAN)],
        target_rows=2000,
        directives=directives or {},
    )


def test_directive_free_shard_bytes_unchanged() -> None:
    a = generate_shard_bytes(_spec(), 0, 200, seed=3)
    b = generate_shard_bytes(_spec(), 0, 200, seed=3)
    assert a == b  # deterministic, unchanged from C0 behavior


def test_shard_bytes_reflect_bias_directive() -> None:
    directive = GenerationDirective(kind=DirectiveKind.BIAS, field="flag", value=True, rate=0.95)
    spec = _spec(dump_directives([directive]))
    data = generate_shard_bytes(spec, 0, 2000, seed=9)
    table = pq.read_table(io.BytesIO(data))
    rate = sum(table.column("flag").to_pylist()) / 2000
    assert rate >= 0.9
