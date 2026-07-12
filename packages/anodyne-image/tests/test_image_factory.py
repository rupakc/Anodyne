from __future__ import annotations

from uuid import uuid4

import pytest
from anodyne_core.models import ModelConfig
from anodyne_image.errors import ImageProviderError
from anodyne_image.factory import register_provider, resolve_image_provider
from anodyne_image.providers.openai import OpenAIImageProvider
from anodyne_image.providers.selfhosted import SelfHostedSDXLProvider


def _config(provider: str, model: str = "m") -> ModelConfig:
    return ModelConfig(id=uuid4(), tenant_id=uuid4(), name="n", provider=provider, model=model)


def test_resolves_openai_images() -> None:
    provider = resolve_image_provider(_config("openai-images", "dall-e-3"), api_key="sk-x")
    assert isinstance(provider, OpenAIImageProvider)


def test_resolves_self_hosted_sdxl() -> None:
    provider = resolve_image_provider(_config("sdxl-self-hosted"), api_key=None)
    assert isinstance(provider, SelfHostedSDXLProvider)


def test_unknown_provider_raises() -> None:
    with pytest.raises(ImageProviderError, match="unknown image provider"):
        resolve_image_provider(_config("carrier-pigeon"), api_key=None)


def test_register_provider_extends_the_registry() -> None:
    class _Custom(SelfHostedSDXLProvider):
        pass

    register_provider("custom-test-provider", lambda cfg, key: _Custom())
    try:
        provider = resolve_image_provider(_config("custom-test-provider"), api_key=None)
        assert isinstance(provider, _Custom)
    finally:
        # Don't leak the test registration into other tests' factory state.
        from anodyne_image.factory import _REGISTRY

        del _REGISTRY["custom-test-provider"]
