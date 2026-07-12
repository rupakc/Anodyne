from __future__ import annotations

from collections.abc import Callable

from anodyne_core.models import ModelConfig

from anodyne_image.errors import ImageProviderError
from anodyne_image.ports import ImageProvider
from anodyne_image.providers.openai import OpenAIImageProvider
from anodyne_image.providers.selfhosted import SelfHostedSDXLProvider

ProviderCtor = Callable[[ModelConfig, str | None], ImageProvider]

# Keyed on `ModelConfig.provider`. A small, explicit registry (rather than a
# hardcoded if/elif) so additional adapters -- Replicate, fal.ai, an
# organization's own GPU pipeline -- register without touching this module,
# per the "provider-agnostic" architecture decision.
_REGISTRY: dict[str, ProviderCtor] = {}


def register_provider(name: str, ctor: ProviderCtor) -> None:
    """Register (or override, e.g. in tests) the adapter constructed for `name`."""
    _REGISTRY[name] = ctor


def resolve_image_provider(config: ModelConfig, api_key: str | None) -> ImageProvider:
    """Build the `ImageProvider` bound to `config` (already decrypted `api_key`, if any).

    Raises `ImageProviderError` for an unregistered `config.provider` name.
    """
    ctor = _REGISTRY.get(config.provider)
    if ctor is None:
        raise ImageProviderError(
            f"unknown image provider: {config.provider!r} (registered: "
            f"{sorted(_REGISTRY)}); register an adapter via "
            "anodyne_image.factory.register_provider"
        )
    return ctor(config, api_key)


register_provider(
    "openai-images",
    lambda cfg, key: OpenAIImageProvider(
        api_key=key,
        model=cfg.model,
        api_base=cfg.api_base or "https://api.openai.com/v1",
        params=cfg.params,
    ),
)
register_provider("sdxl-self-hosted", lambda cfg, key: SelfHostedSDXLProvider())
