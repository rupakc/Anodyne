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


def ray_init(address: str) -> None:
    """Initialize Ray for this process, connecting to a remote cluster if configured.

    Args:
        address: Ray cluster address to connect to, e.g. "ray://host:10001".
            If falsy (e.g. ``""``), a local Ray instance is started instead
            (single-process dev/test runs). Idempotent: a no-op if Ray is
            already initialized in this process (guarded on
            `ray.is_initialized()`), so it's safe to call at worker startup
            without racing a prior/concurrent initialization.
    """
    if ray.is_initialized():
        return
    if address:
        ray.init(address=address)
    else:
        ray.init()
