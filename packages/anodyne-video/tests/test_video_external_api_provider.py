import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from anodyne_core.ports import SecretStore
from anodyne_video.adapters.external_api import ReplicateVideoProvider
from anodyne_video.models import VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProviderError


class _FakeSecretStore(SecretStore):
    def encrypt(self, plaintext: str) -> str:
        return f"enc:{plaintext}"

    def decrypt(self, ref: str) -> str:
        return ref.removeprefix("enc:")


def _config(**overrides: object) -> VideoProviderConfig:
    defaults: dict[str, object] = dict(
        id=uuid4(),
        tenant_id=uuid4(),
        name="replicate",
        provider="replicate",
        model="stability-ai/stable-video-diffusion",
        secret_ref="enc:sk-test-123",
    )
    defaults.update(overrides)
    return VideoProviderConfig.model_validate(defaults)


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_generate_creates_prediction_polls_and_downloads_output() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v1/predictions" and request.method == "POST":
            return httpx.Response(201, json={"id": "pred-1", "status": "starting"})
        if request.url.path == "/v1/predictions/pred-1" and request.method == "GET":
            if sum(1 for c in calls if c.url.path == "/v1/predictions/pred-1") == 1:
                return httpx.Response(200, json={"id": "pred-1", "status": "processing"})
            return httpx.Response(
                200,
                json={
                    "id": "pred-1",
                    "status": "succeeded",
                    "output": "https://replicate.delivery/output.mp4",
                },
            )
        if str(request.url) == "https://replicate.delivery/output.mp4":
            return httpx.Response(200, content=b"fake-mp4-bytes")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    provider = ReplicateVideoProvider(secret_store=_FakeSecretStore(), client=_client(handler))
    config = _config()
    request = VideoGenerationRequest(prompt="a cat surfing", seed=1)

    asset = await provider.generate(config, request)

    assert asset.content == b"fake-mp4-bytes"
    assert asset.provider == "replicate"
    assert asset.model == config.model

    create_call = next(c for c in calls if c.method == "POST")
    assert create_call.headers["authorization"] == "Bearer sk-test-123"
    body = json.loads(create_call.content)
    assert body["input"]["prompt"] == "a cat surfing"
    assert body["version"] == config.model


async def test_generate_raises_video_provider_error_on_failed_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": "pred-1", "status": "starting"})
        return httpx.Response(
            200, json={"id": "pred-1", "status": "failed", "error": "model exploded"}
        )

    provider = ReplicateVideoProvider(secret_store=_FakeSecretStore(), client=_client(handler))

    with pytest.raises(VideoProviderError, match="model exploded"):
        await provider.generate(_config(), VideoGenerationRequest(prompt="x", seed=1))


async def test_generate_bounds_polling_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": "pred-1", "status": "starting"})
        return httpx.Response(200, json={"id": "pred-1", "status": "processing"})

    provider = ReplicateVideoProvider(
        secret_store=_FakeSecretStore(),
        client=_client(handler),
        max_poll_attempts=2,
        poll_interval_seconds=0,
    )

    with pytest.raises(VideoProviderError, match="did not complete"):
        await provider.generate(_config(), VideoGenerationRequest(prompt="x", seed=1))
