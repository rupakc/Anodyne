"""Ray dispatch for the from-sample path: sample a *pre-fit* `Generator`, once per shard.

Unlike `ray_tasks.generate_shard_bytes` (which always constructs a fresh `TabularSampler`),
these tasks take an already-fitted `Generator` instance (a `CopulaTabularGenerator`,
`DeepTabularGenerator`, or `SdvGaussianCopulaGenerator` from `anodyne_tabular`) and ship it to
the Ray worker, since fitting a statistical/deep model per shard would be wasteful and would
break the seed-determinism contract.
"""

from __future__ import annotations

import io

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import Generator


def generate_shard_bytes_from_generator(
    generator: Generator, spec: DatasetSpec, start_row: int, count: int, seed: int
) -> bytes:
    """Generate Parquet bytes for a data shard using an already-fitted `generator`."""
    table = generator.generate(spec, start_row, count, seed)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@ray.remote
def remote_generate_shard_from_generator(
    generator: Generator, spec: DatasetSpec, start_row: int, count: int, seed: int
) -> bytes:
    """Ray remote task: generate a data shard using an already-fitted `generator`."""
    return generate_shard_bytes_from_generator(generator, spec, start_row, count, seed)
