from __future__ import annotations

from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider
from anodyne_dataset.models import AudioSynthesisRequest


async def test_delegates_to_injected_synthesize_fn() -> None:
    calls: list[tuple[str, str | None]] = []

    async def fake_synthesize(text: str, voice: str | None) -> bytes:
        calls.append((text, voice))
        return b"pcm-bytes"

    provider = SelfHostedAudioProvider(fake_synthesize, model_name="xtts_v2")

    result = await provider.synthesize(AudioSynthesisRequest(text="hi", voice="narrator"))

    assert result.audio_bytes == b"pcm-bytes"
    assert result.format == "wav"
    assert calls == [("hi", "narrator")]


async def test_default_format_is_overridable() -> None:
    async def fake_synthesize(text: str, voice: str | None) -> bytes:
        return b"x"

    provider = SelfHostedAudioProvider(fake_synthesize, format="pcm16")

    result = await provider.synthesize(AudioSynthesisRequest(text="t"))

    assert result.format == "pcm16"


async def test_passes_no_voice_when_request_has_none() -> None:
    calls: list[tuple[str, str | None]] = []

    async def fake_synthesize(text: str, voice: str | None) -> bytes:
        calls.append((text, voice))
        return b""

    provider = SelfHostedAudioProvider(fake_synthesize)

    await provider.synthesize(AudioSynthesisRequest(text="no voice"))

    assert calls == [("no voice", None)]
