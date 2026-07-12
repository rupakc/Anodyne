from __future__ import annotations

from collections.abc import Awaitable, Callable

from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult
from anodyne_dataset.ports import AudioProvider

SynthesizeFn = Callable[[str, str | None], Awaitable[bytes]]


class SelfHostedAudioProvider(AudioProvider):
    """Adapter for self-hosted OSS TTS/audio models (e.g. XTTS, Bark) served on
    a Ray GPU actor.

    This adapter has no direct Ray/GPU dependency itself: it delegates to an
    injected `synthesize_fn`. Production wiring (`apps/generation-worker`)
    supplies one that calls a Ray remote GPU actor --
    `anodyne_compute.audio_actor.SelfHostedTTSActor` -- which requires a GPU
    node pool and the target model package; neither is available in this
    environment, so unit tests inject a plain async fake instead.
    """

    def __init__(
        self,
        synthesize_fn: SynthesizeFn,
        *,
        format: str = "wav",
        model_name: str = "self-hosted-tts",
    ) -> None:
        self._synthesize_fn = synthesize_fn
        self._format = format
        self._model_name = model_name

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        audio_bytes = await self._synthesize_fn(request.text, request.voice)
        return AudioSynthesisResult(audio_bytes=audio_bytes, format=self._format)
