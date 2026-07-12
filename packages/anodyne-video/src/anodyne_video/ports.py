from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig


class VideoProviderError(Exception):
    """Raised when a `VideoProvider` fails to produce a `VideoAsset`."""


class VideoProvider(ABC):
    """Generates one video clip from a prompt. One adapter per provider (self-hosted
    GPU model or external API); selected per tenant via `VideoProviderConfig.provider`.
    """

    @abstractmethod
    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset: ...


class VideoProviderRegistry(ABC):
    """Tenant-scoped CRUD over `VideoProviderConfig`. Mirrors the shape of
    `anodyne_llm.registry.SqlModelRegistry` (there declared informally as a
    `Protocol` in the gateway; declared here as a formal port since this is a
    fresh package).
    """

    @abstractmethod
    async def create(
        self,
        tenant_id: UUID,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> VideoProviderConfig: ...

    @abstractmethod
    async def get(self, tenant_id: UUID, config_id: UUID) -> VideoProviderConfig | None: ...

    @abstractmethod
    async def list(self, tenant_id: UUID) -> list[VideoProviderConfig]: ...

    @abstractmethod
    async def delete(self, tenant_id: UUID, config_id: UUID) -> None: ...
