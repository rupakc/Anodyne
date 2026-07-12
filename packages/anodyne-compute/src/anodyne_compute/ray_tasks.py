from __future__ import annotations

import io

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_dataset.models import DatasetSpec
from anodyne_generation.sampler import TabularSampler


def generate_shard_bytes(spec: DatasetSpec, start_row: int, count: int, seed: int) -> bytes:
    """Generate Parquet bytes for a data shard.

    Args:
        spec: Dataset specification defining the schema and generation parameters.
        start_row: Starting row index for this shard.
        count: Number of rows to generate.
        seed: Random seed for reproducible generation.

    Returns:
        Parquet-encoded bytes for the generated shard.
    """
    table = TabularSampler().generate(spec, start_row, count, seed)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@ray.remote
def remote_generate_shard(spec: DatasetSpec, start_row: int, count: int, seed: int) -> bytes:
    """Ray remote task for generating a data shard.

    Args:
        spec: Dataset specification defining the schema and generation parameters.
        start_row: Starting row index for this shard.
        count: Number of rows to generate.
        seed: Random seed for reproducible generation.

    Returns:
        Parquet-encoded bytes for the generated shard.
    """
    return generate_shard_bytes(spec, start_row, count, seed)


def ray_init(address: str | None) -> None:
    """Initialize Ray cluster if not already initialized.

    Args:
        address: Ray cluster address. If None or "auto", uses default.
                 Pass None for local mode in tests.
    """
    if not ray.is_initialized():
        ray.init(address=address or "auto", ignore_reinit_error=True)
