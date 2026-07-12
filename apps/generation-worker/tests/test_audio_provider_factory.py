"""Tests for `generation_worker.audio.AudioProviderFactory`.

All against fake registries/secrets -- no live DB, no Ray, no network. Proves
the tenant-`ModelConfig` -> concrete-adapter mapping described in the C4 spec
(reusing the `anodyne-llm` model-registry pattern for audio providers).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider
from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider
from anodyne_core.models import ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from generation_worker.audio import AudioProviderFactory


class _FakeSecrets(SecretStore):
    def encrypt(self, plaintext: str) -> str:
        return f"enc:{plaintext}"

    def decrypt(self, ref: str) -> str:
        return ref.removeprefix("enc:")


class _FakeRegistry:
    def __init__(self, configs: list[ModelConfig]) -> None:
        self._configs = configs

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        return next((c for c in self._configs if c.id == config_id), None)

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        return [c for c in self._configs if c.tenant_id == tenant_id]


def _spec(tenant_id: UUID, directives: dict[str, object] | None = None) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=tenant_id,
        name="d",
        description="",
        modality=Modality.AUDIO,
        source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=1,
        directives=directives or {},
    )


async def test_builds_elevenlabs_provider_for_elevenlabs_config() -> None:
    tid = uuid4()
    cfg = ModelConfig(
        id=uuid4(),
        tenant_id=tid,
        name="m",
        provider="elevenlabs",
        model="voice-1",
        secret_ref="enc:sk-live",
    )
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())

    provider = await factory.build(_spec(tid))

    assert isinstance(provider, ElevenLabsAudioProvider)


async def test_builds_selfhosted_provider_for_other_providers() -> None:
    tid = uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tid, name="m", provider="xtts", model="xtts_v2")
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())

    provider = await factory.build(_spec(tid))

    assert isinstance(provider, SelfHostedAudioProvider)


async def test_prefers_explicit_model_config_id_from_directives() -> None:
    tid = uuid4()
    wanted = ModelConfig(
        id=uuid4(), tenant_id=tid, name="wanted", provider="elevenlabs", model="v2"
    )
    other = ModelConfig(id=uuid4(), tenant_id=tid, name="other", provider="xtts", model="x")
    factory = AudioProviderFactory(_FakeRegistry([other, wanted]), _FakeSecrets())

    provider = await factory.build(_spec(tid, {"audio": {"model_config_id": str(wanted.id)}}))

    assert isinstance(provider, ElevenLabsAudioProvider)


async def test_falls_back_to_first_config_if_directive_id_not_found() -> None:
    tid = uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tid, name="m", provider="elevenlabs", model="v1")
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())

    provider = await factory.build(_spec(tid, {"audio": {"model_config_id": str(uuid4())}}))

    assert isinstance(provider, ElevenLabsAudioProvider)


async def test_raises_when_no_audio_provider_configured() -> None:
    tid = uuid4()
    factory = AudioProviderFactory(_FakeRegistry([]), _FakeSecrets())

    with pytest.raises(RuntimeError, match="no audio provider configured"):
        await factory.build(_spec(tid))


async def test_elevenlabs_provider_decrypts_secret() -> None:
    tid = uuid4()
    cfg = ModelConfig(
        id=uuid4(),
        tenant_id=tid,
        name="m",
        provider="elevenlabs",
        model="v1",
        secret_ref="enc:sk-super-secret",
    )
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())

    provider = await factory.build(_spec(tid))

    assert isinstance(provider, ElevenLabsAudioProvider)
    assert provider._api_key == "sk-super-secret"  # noqa: SLF001 - white-box check


async def test_elevenlabs_provider_without_secret_ref_uses_empty_api_key() -> None:
    tid = uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tid, name="m", provider="elevenlabs", model="v1")
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())

    provider = await factory.build(_spec(tid))

    assert isinstance(provider, ElevenLabsAudioProvider)
    assert provider._api_key == ""  # noqa: SLF001 - white-box check
