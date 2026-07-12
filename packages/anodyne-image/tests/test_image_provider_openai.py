from __future__ import annotations

import base64
from typing import Any

import pytest
from anodyne_image.errors import ImageProviderError
from anodyne_image.providers.openai import OpenAIImageProvider


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict[str, Any], text: str = "") -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text or str(json_body)

    def json(self) -> dict[str, Any]:
        return self._json


class _FakeHttpClient:
    """Records the call it received and returns a pre-baked response -- no network."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,  # noqa: ASYNC109 -- mirrors httpx.AsyncClient.post's real signature
    ) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self._response


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def test_generate_decodes_b64_image() -> None:
    image_bytes = b"\x89PNGfake"
    client = _FakeHttpClient(_FakeResponse(200, {"data": [{"b64_json": _b64(image_bytes)}]}))
    provider = OpenAIImageProvider(
        api_key="sk-test", model="dall-e-3", http_client=client, params={"quality": "hd"}
    )

    result = await provider.generate("a cat", seed=1, size="512x512")

    assert result.data == image_bytes
    assert result.mime_type == "image/png"


async def test_request_shape() -> None:
    client = _FakeHttpClient(_FakeResponse(200, {"data": [{"b64_json": _b64(b"x")}]}))
    provider = OpenAIImageProvider(api_key="sk-test", model="dall-e-3", http_client=client)

    await provider.generate("a dog", seed=2, size="256x256")

    call = client.calls[0]
    assert call["url"] == "https://api.openai.com/v1/images/generations"
    assert call["json"]["model"] == "dall-e-3"
    assert call["json"]["prompt"] == "a dog"
    assert call["json"]["size"] == "256x256"
    assert call["json"]["n"] == 1
    assert call["json"]["response_format"] == "b64_json"
    assert call["headers"]["Authorization"] == "Bearer sk-test"


async def test_custom_api_base_used() -> None:
    client = _FakeHttpClient(_FakeResponse(200, {"data": [{"b64_json": _b64(b"x")}]}))
    provider = OpenAIImageProvider(
        api_key="k", model="m", http_client=client, api_base="https://proxy.example.com/v1"
    )

    await provider.generate("p", seed=1)

    assert client.calls[0]["url"] == "https://proxy.example.com/v1/images/generations"


async def test_non_2xx_raises_with_body() -> None:
    client = _FakeHttpClient(_FakeResponse(429, {"error": "rate limited"}, text="rate limited"))
    provider = OpenAIImageProvider(api_key="k", model="m", http_client=client)

    with pytest.raises(ImageProviderError, match="429"):
        await provider.generate("p", seed=1)


async def test_missing_b64_json_raises() -> None:
    client = _FakeHttpClient(_FakeResponse(200, {"data": [{}]}))
    provider = OpenAIImageProvider(api_key="k", model="m", http_client=client)

    with pytest.raises(ImageProviderError, match="b64_json"):
        await provider.generate("p", seed=1)


async def test_missing_api_key_raises_before_any_call() -> None:
    client = _FakeHttpClient(_FakeResponse(200, {"data": [{"b64_json": _b64(b"x")}]}))
    provider = OpenAIImageProvider(api_key=None, model="m", http_client=client)

    with pytest.raises(ImageProviderError, match="API key"):
        await provider.generate("p", seed=1)

    assert client.calls == []
