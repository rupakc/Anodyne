from __future__ import annotations

import httpx
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult
from anodyne_dataset.ports import AudioProvider

_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsError(Exception):
    """Raised when the ElevenLabs API returns a non-2xx response."""


class ElevenLabsAudioProvider(AudioProvider):
    """External-API adapter for ElevenLabs text-to-speech.

    `POST /v1/text-to-speech/{voice_id}` with a JSON body of `{text, model_id}`
    and an `xi-api-key` header; the response body is the raw audio bytes. A
    request's own `voice` overrides the adapter's default `voice_id`, so a
    single DatasetSpec can mix voices per item via
    `directives["audio"]["voice"]` while still having a sane tenant-level
    default configured on the `ModelConfig`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._client = http_client or httpx.AsyncClient()

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        voice_id = request.voice or self._voice_id
        resp = await self._client.post(
            f"{_BASE_URL}/{voice_id}",
            json={"text": request.text, "model_id": self._model_id},
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise ElevenLabsError(f"ElevenLabs TTS failed ({resp.status_code}): {resp.text}")
        content_type = resp.headers.get("content-type", "audio/mpeg")
        subtype = content_type.split(";", 1)[0].split("/")[-1]
        fmt = "mp3" if subtype == "mpeg" else subtype
        return AudioSynthesisResult(audio_bytes=resp.content, format=fmt)
