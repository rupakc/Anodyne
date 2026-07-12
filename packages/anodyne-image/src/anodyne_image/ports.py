from __future__ import annotations

from abc import ABC, abstractmethod

from anodyne_image.models import GeneratedImage


class ImageProvider(ABC):
    """Provider-agnostic image generation port.

    One instance is already bound to a single tenant's chosen model/key/base-url
    (constructed via `anodyne_image.factory.resolve_image_provider`) -- unlike
    `anodyne_core.ports.LLMProvider.complete(config, request)`, which re-selects
    a model per call. Images don't need per-call provider switching, so binding
    at construction keeps adapters (and their tests) simpler.

    Implementations: `anodyne_image.providers.openai.OpenAIImageProvider`
    (external API) and `anodyne_image.providers.selfhosted.SelfHostedSDXLProvider`
    (self-hosted GPU OSS model, e.g. SDXL served via a Ray GPU actor).
    """

    @abstractmethod
    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        """Generate one image for `prompt`.

        `seed` should make repeated calls reproducible where the underlying
        model/API supports seeding; callers must not assume determinism across
        providers that don't (e.g. some external APIs ignore seeds).
        """
