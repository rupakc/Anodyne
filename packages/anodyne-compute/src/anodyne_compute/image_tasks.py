from __future__ import annotations

import io

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import DatasetSpec
from anodyne_image.factory import resolve_image_provider
from anodyne_image.generator import ImageGenerator


def generate_image_shard_bytes(
    spec: DatasetSpec,
    start_row: int,
    count: int,
    seed: int,
    provider_config: ModelConfig,
    api_key: str | None,
) -> bytes:
    """Generate Parquet bytes for an image shard.

    `provider_config` (a picklable `ModelConfig`) and `api_key` (the already
    -decrypted plaintext, also picklable) are passed rather than a
    constructed `ImageProvider`/`SecretStore` -- these are exactly the two
    plain, serializable values that need to cross the Ray worker-process
    boundary; the provider itself is (re)built inside this call via
    `resolve_image_provider`, mirroring `ray_tasks.generate_shard_bytes`'s
    picklable-args pattern for the tabular path.
    """
    provider = resolve_image_provider(provider_config, api_key)
    table = ImageGenerator(provider).generate(spec, start_row, count, seed)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@ray.remote
def remote_generate_image_shard(
    spec: DatasetSpec,
    start_row: int,
    count: int,
    seed: int,
    provider_config: ModelConfig,
    api_key: str | None,
) -> bytes:
    """Ray remote task for generating an image shard.

    Self-hosted GPU providers resolve to a `SelfHostedSDXLProvider` here;
    live inference additionally requires this task to run on (or reach, via
    a `RayGpuActorPipeline`) a GPU node -- not available in this environment,
    so tests only ever register a fake provider (see
    `tests/test_image_ray_tasks.py`).
    """
    return generate_image_shard_bytes(spec, start_row, count, seed, provider_config, api_key)
