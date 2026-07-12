from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider, ElevenLabsError
from anodyne_dataset.models import AudioSynthesisRequest


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_posts_expected_url_headers_and_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=b"RIFF...audio...", headers={"content-type": "audio/mpeg"}
        )

    provider = ElevenLabsAudioProvider(
        api_key="sk-test", voice_id="v1", http_client=_client(handler)
    )

    result = await provider.synthesize(AudioSynthesisRequest(text="hello world"))

    assert captured["url"] == "https://api.elevenlabs.io/v1/text-to-speech/v1"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["xi-api-key"] == "sk-test"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["text"] == "hello world"
    assert body["model_id"] == "eleven_multilingual_v2"
    assert result.audio_bytes == b"RIFF...audio..."
    assert result.format == "mp3"


async def test_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid api key"})

    provider = ElevenLabsAudioProvider(api_key="bad", voice_id="v1", http_client=_client(handler))

    with pytest.raises(ElevenLabsError):
        await provider.synthesize(AudioSynthesisRequest(text="x"))


async def test_voice_override_uses_request_voice_not_default() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"abc")

    provider = ElevenLabsAudioProvider(
        api_key="k", voice_id="default-voice", http_client=_client(handler)
    )

    await provider.synthesize(AudioSynthesisRequest(text="hi", voice="override-voice"))

    assert seen["url"].endswith("/override-voice")


async def test_custom_model_id_is_used_in_request_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"abc")

    provider = ElevenLabsAudioProvider(
        api_key="k", voice_id="v1", model_id="eleven_turbo_v2", http_client=_client(handler)
    )

    await provider.synthesize(AudioSynthesisRequest(text="hi"))

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model_id"] == "eleven_turbo_v2"


async def test_unknown_content_type_falls_back_to_subtype_as_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"abc", headers={"content-type": "audio/wav"})

    provider = ElevenLabsAudioProvider(api_key="k", voice_id="v1", http_client=_client(handler))

    result = await provider.synthesize(AudioSynthesisRequest(text="hi"))

    assert result.format == "wav"
