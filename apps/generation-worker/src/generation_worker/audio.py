"""Builds the real per-tenant `AudioProvider` for `GenerationWorkflow`'s audio path.

Maps a tenant's registered `ModelConfig.provider` to the concrete adapter:
- `"elevenlabs"` -> `ElevenLabsAudioProvider` (external API; needs a registered
  API key, decrypted via the same `SecretStore` used for LLM model configs).
- anything else (e.g. `"xtts"`, `"bark"`, `"selfhosted"`) -> `SelfHostedAudioProvider`,
  wired to a lazily-created Ray actor handle (`anodyne_compute.audio_actor.SelfHostedTTSActor`)
  -- requires a GPU node pool and the target model package; not exercised
  without one (Ray/GPU imports are deferred so building the factory itself
  never touches Ray).

Selection: `spec.directives["audio"]["model_config_id"]` if set and found,
else the tenant's first registered audio-provider config. The registry passed
in is `anodyne_audio.registry.SqlAudioProviderRegistry` (over the dedicated
`audio_provider_configs` table, encrypted-secret pattern identical to the LLM
model registry) -- so audio stores its providers the same way image/video do,
rather than reusing the LLM `model_configs` table.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol
from uuid import UUID

from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider
from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider
from anodyne_core.models import ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import AudioProvider

_ELEVENLABS = "elevenlabs"


class ModelRegistryLike(Protocol):
    """Structural type for the registry consumed by `AudioProviderFactory`.

    `anodyne_llm.registry.SqlModelRegistry` is the real implementation; tests
    substitute an in-memory fake.
    """

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None: ...

    async def list(self, tenant_id: UUID) -> list[ModelConfig]: ...


def _audio_directives(spec: DatasetSpec) -> dict[str, object]:
    raw = spec.directives.get("audio")
    return raw if isinstance(raw, dict) else {}


class AudioProviderFactory:
    """Resolves a tenant's registered `ModelConfig` into a concrete `AudioProvider`."""

    def __init__(self, registry: ModelRegistryLike, secrets: SecretStore) -> None:
        self._registry = registry
        self._secrets = secrets
        self._actor: Any | None = None

    async def build(self, spec: DatasetSpec) -> AudioProvider:
        cfg = await self._resolve_config(spec)
        if cfg.provider == _ELEVENLABS:
            api_key = self._secrets.decrypt(cfg.secret_ref) if cfg.secret_ref else ""
            return ElevenLabsAudioProvider(api_key=api_key, voice_id=cfg.model)
        return SelfHostedAudioProvider(self._ray_synthesize, model_name=cfg.model)

    async def _resolve_config(self, spec: DatasetSpec) -> ModelConfig:
        directives = _audio_directives(spec)
        config_id = directives.get("model_config_id")
        if config_id:
            cfg = await self._registry.get(spec.tenant_id, UUID(str(config_id)))
            if cfg is not None:
                return cfg
        configs = await self._registry.list(spec.tenant_id)
        if not configs:
            raise RuntimeError(f"no audio provider configured for tenant {spec.tenant_id}")
        return configs[0]

    def _actor_handle(self) -> Any:
        if self._actor is None:
            from anodyne_compute.audio_actor import SelfHostedTTSActor

            # requires a GPU node pool + Ray; `.remote` is added dynamically by
            # `@ray.remote`, which ray's stubs don't model on actor classes.
            self._actor = SelfHostedTTSActor.remote()  # type: ignore[attr-defined]
        return self._actor

    async def _ray_synthesize(self, text: str, voice: str | None) -> bytes:
        import ray

        ref = self._actor_handle().synthesize.remote(text, voice)
        return await asyncio.to_thread(ray.get, ref)
