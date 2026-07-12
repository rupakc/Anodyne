from __future__ import annotations

import base64
from typing import Any, Protocol, cast

from anodyne_image.errors import ImageProviderError
from anodyne_image.models import GeneratedImage
from anodyne_image.ports import ImageProvider

_DEFAULT_API_BASE = "https://api.openai.com/v1"
_TIMEOUT_SECONDS = 120.0


class _HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> dict[str, Any]: ...


class _HttpClient(Protocol):
    """Duck-typed async HTTP POST client.

    `httpx.AsyncClient` satisfies this structurally, but the port is defined
    narrowly (just the one method this adapter needs) so tests can inject a
    trivial fake instead of a mocked `httpx` transport -- no live network, no
    extra mocking dependency.
    """

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,  # noqa: ASYNC109 -- mirrors httpx.AsyncClient.post's real signature
    ) -> _HttpResponse: ...


class OpenAIImageProvider(ImageProvider):
    """External-API adapter: OpenAI's `POST /images/generations`.

    Bound to one tenant's model/key/base-url at construction (see
    `anodyne_image.ports.ImageProvider`). Requests `response_format=b64_json`
    so the image bytes come back inline (no second fetch of a signed URL).

    Live calls require a real API key registered for the tenant (via
    `/image-providers`) -- there are none in this environment; every test
    injects a fake `http_client`.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        api_base: str = _DEFAULT_API_BASE,
        params: dict[str, object] | None = None,
        http_client: _HttpClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._params = params or {}
        self._http_client = http_client

    def _client(self) -> _HttpClient:
        if self._http_client is not None:
            return self._http_client
        import httpx

        # httpx.AsyncClient.post's real signature is broader than `_HttpClient`
        # (it accepts far more than json/headers/timeout); the cast just
        # asserts it's a superset. Never exercised in tests -- no network here.
        return cast(
            "_HttpClient", httpx.AsyncClient()
        )  # pragma: no cover - real network path, never hit in tests

    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        if not self._api_key:
            raise ImageProviderError(
                "OpenAI image provider requires an API key; register one via "
                "POST /image-providers before generating."
            )
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
            **self._params,
        }
        resp = await self._client().post(
            f"{self._api_base}/images/generations",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            raise ImageProviderError(f"OpenAI image API returned {resp.status_code}: {resp.text}")
        data = resp.json().get("data") or []
        if not data or not data[0].get("b64_json"):
            raise ImageProviderError(
                "OpenAI image API response missing expected 'data[0].b64_json' field"
            )
        return GeneratedImage(data=base64.b64decode(data[0]["b64_json"]), mime_type="image/png")
