from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, TenantContext


class ObjectStore(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def presigned_url(self, key: str, expires: int = 3600) -> str: ...

    @abstractmethod
    async def list(self, prefix: str) -> list[str]: ...


class SecretStore(ABC):
    @abstractmethod
    def encrypt(self, plaintext: str) -> str: ...

    @abstractmethod
    def decrypt(self, ref: str) -> str: ...


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]: ...


class AuthorizationPolicy(ABC):
    @abstractmethod
    def is_permitted(self, ctx: TenantContext, permission: str) -> bool: ...
