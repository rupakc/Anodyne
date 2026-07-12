"""External-API `VideoProvider` adapter: Replicate's REST shape.

Three-step flow (the shape most hosted-inference video APIs share, Replicate
included -- fal.ai/Runway differ in wire format but fit the same shape behind
their own adapter, a follow-up rather than a redesign):

1. ``POST {api_base}/predictions`` with the model version + input -> a
   prediction id, initially ``status: "starting"``.
2. Poll ``GET {api_base}/predictions/{id}`` until ``status`` reaches a
   terminal state (``"succeeded"`` or ``"failed"``), bounded by
   ``max_poll_attempts`` so a stuck provider fails fast instead of hanging
   forever.
3. On success, download the ``output`` URL's bytes as the clip content.

No network call in this module's tests: `httpx.AsyncClient` is injected, and
tests wire it to an in-process `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio

import httpx
from anodyne_core.ports import SecretStore

from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProvider, VideoProviderError

_DEFAULT_API_BASE = "https://api.replicate.com/v1"
_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


class ReplicateVideoProvider(VideoProvider):
    def __init__(
        self,
        *,
        secret_store: SecretStore,
        client: httpx.AsyncClient,
        max_poll_attempts: int = 30,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._secrets = secret_store
        self._client = client
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval_seconds = poll_interval_seconds

    def _api_base(self, config: VideoProviderConfig) -> str:
        return config.api_base or _DEFAULT_API_BASE

    def _headers(self, config: VideoProviderConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.secret_ref:
            headers["Authorization"] = f"Bearer {self._secrets.decrypt(config.secret_ref)}"
        return headers

    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset:
        base = self._api_base(config)
        headers = self._headers(config)
        payload: dict[str, object] = {
            "version": config.model,
            "input": {
                "prompt": request.prompt,
                "duration_seconds": request.duration_seconds,
                "width": request.width,
                "height": request.height,
                "fps": request.fps,
                "seed": request.seed,
                **request.params,
            },
        }
        resp = await self._client.post(f"{base}/predictions", headers=headers, json=payload)
        resp.raise_for_status()
        prediction_id = resp.json()["id"]

        prediction = await self._poll(base, headers, prediction_id)
        if prediction["status"] == "failed":
            raise VideoProviderError(
                f"video generation failed: {prediction.get('error', 'unknown error')}"
            )

        output_url = str(prediction["output"])
        content_resp = await self._client.get(output_url)
        content_resp.raise_for_status()

        return VideoAsset(
            content=content_resp.content,
            duration_seconds=request.duration_seconds,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            provider=config.provider,
            model=config.model,
        )

    async def _poll(
        self, base: str, headers: dict[str, str], prediction_id: str
    ) -> dict[str, object]:
        url = f"{base}/predictions/{prediction_id}"
        for attempt in range(self._max_poll_attempts):
            resp = await self._client.get(url, headers=headers)
            resp.raise_for_status()
            prediction: dict[str, object] = resp.json()
            if prediction["status"] in _TERMINAL_STATUSES:
                return prediction
            if attempt + 1 < self._max_poll_attempts and self._poll_interval_seconds:
                await asyncio.sleep(self._poll_interval_seconds)
        raise VideoProviderError(
            f"video generation did not complete within {self._max_poll_attempts} poll attempts"
        )
