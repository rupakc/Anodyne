from datetime import UTC, datetime
from uuid import uuid4

from anodyne_video.models import (
    VideoAsset,
    VideoGenerationRequest,
    VideoManifest,
    VideoManifestItem,
    VideoProviderConfig,
)


def test_video_provider_config_defaults() -> None:
    cfg = VideoProviderConfig(
        id=uuid4(), tenant_id=uuid4(), name="c", provider="replicate", model="svd-xt"
    )
    assert cfg.enabled is True
    assert cfg.secret_ref is None
    assert cfg.api_base is None
    assert cfg.params == {}


def test_video_generation_request_defaults() -> None:
    req = VideoGenerationRequest(prompt="a cat surfing", seed=1)
    assert req.duration_seconds == 4.0
    assert req.width == 576
    assert req.height == 320
    assert req.fps == 8
    assert req.params == {}


def test_video_asset_round_trips_content_bytes() -> None:
    asset = VideoAsset(
        content=b"fake-mp4-bytes",
        content_type="video/mp4",
        duration_seconds=4.0,
        width=576,
        height=320,
        fps=8,
        seed=1,
        provider="replicate",
        model="svd-xt",
    )
    assert asset.content == b"fake-mp4-bytes"
    assert asset.content_type == "video/mp4"


def test_video_manifest_item_and_manifest_construct_and_dump() -> None:
    item = VideoManifestItem(
        index=0,
        object_key="datasets/d/j/videos/item-0.mp4",
        prompt="a cat surfing",
        duration_seconds=4.0,
        width=576,
        height=320,
        fps=8,
        seed=1,
        provider="replicate",
        model="svd-xt",
        content_type="video/mp4",
        byte_size=14,
    )
    manifest = VideoManifest(
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        job_id=uuid4(),
        items=[item],
    )
    assert manifest.items[0].object_key.endswith("item-0.mp4")
    assert manifest.created_at.tzinfo is not None
    dumped = manifest.model_dump(mode="json")
    assert dumped["items"][0]["index"] == 0
    assert isinstance(datetime.fromisoformat(dumped["created_at"]), datetime)
    assert manifest.created_at <= datetime.now(UTC)
