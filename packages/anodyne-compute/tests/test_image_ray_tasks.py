"""Mirrors `test_ray_tasks.py` for the image-modality shard task -- no GPU, no
live API. Ray workers here are separate OS processes (each with its own
fresh Python import of `anodyne_image`), so a provider registered at test
runtime via `register_provider()` in the driver process is invisible inside
the worker; the happy-path/parquet-shape test below stays local (no Ray
boundary) with a fake provider, and the Ray-remote test proves the
plumbing (spec/config pickle across the process boundary, same code path)
using the always-available built-in `"sdxl-self-hosted"` provider, which
deterministically raises the same, clear error locally and remotely since
no GPU pipeline is configured in this environment.
"""

from __future__ import annotations

import io
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import ray
from anodyne_compute.image_tasks import generate_image_shard_bytes, remote_generate_image_shard
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_image.errors import ImageProviderError
from anodyne_image.factory import _REGISTRY, register_provider
from anodyne_image.models import GeneratedImage
from anodyne_image.ports import ImageProvider

pytestmark = pytest.mark.integration

_FAKE_PROVIDER_NAME = "fake-test-provider"


class _FakeProvider(ImageProvider):
    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        return GeneratedImage(data=f"{prompt}|{seed}".encode())


@pytest.fixture
def fake_provider_config():  # type: ignore[no-untyped-def]
    """Registers a fake provider for the *local, same-process* test only."""
    register_provider(_FAKE_PROVIDER_NAME, lambda cfg, key: _FakeProvider())
    yield ModelConfig(
        id=uuid4(), tenant_id=uuid4(), name="n", provider=_FAKE_PROVIDER_NAME, model="m"
    )
    del _REGISTRY[_FAKE_PROVIDER_NAME]


def _spec() -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="a widget",
        modality=Modality.IMAGE,
        source="description",
        fields=[
            FieldSpec(
                name="label",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["a"]},
            )
        ],
        target_rows=6,
    )


def _self_hosted_config() -> ModelConfig:
    return ModelConfig(
        id=uuid4(), tenant_id=uuid4(), name="n", provider="sdxl-self-hosted", model="sdxl"
    )


def test_generate_image_shard_bytes_is_parquet(fake_provider_config: ModelConfig) -> None:
    data = generate_image_shard_bytes(
        _spec(), 0, 6, seed=3, provider_config=fake_provider_config, api_key=None
    )
    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 6
    assert set(table.column_names) == {"item_index", "label", "prompt", "image_bytes", "mime_type"}


def test_ray_remote_and_local_agree_on_no_gpu_pipeline_configured() -> None:
    """Proves the Ray round-trip: `DatasetSpec`/`ModelConfig` pickle across the
    worker-process boundary and drive the identical `generate_image_shard_bytes`
    code path remotely as locally -- surfaced here as both raising the same
    clear error, since no GPU pipeline is configured in this environment
    (the only outcome that's both deterministic and network/GPU-free).
    """
    spec, config = _spec(), _self_hosted_config()

    with pytest.raises(ImageProviderError, match="GPU"):
        generate_image_shard_bytes(spec, 0, 6, seed=3, provider_config=config, api_key=None)

    ray.init(ignore_reinit_error=True)
    try:
        with pytest.raises(ImageProviderError, match="GPU"):
            ray.get(remote_generate_image_shard.remote(spec, 0, 6, 3, config, None))
    finally:
        ray.shutdown()
