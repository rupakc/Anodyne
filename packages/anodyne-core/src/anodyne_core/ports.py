from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import IO

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, TenantContext


class ObjectStore(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes) -> None: ...

    async def put_fileobj(self, key: str, fileobj: IO[bytes]) -> None:
        """Upload from a binary file object rather than an in-memory `bytes`.

        Concrete (non-abstract) default so existing implementations keep working
        unchanged: it reads the whole object and delegates to `put`. Streaming
        stores (e.g. `S3ObjectStore`) override this to upload directly from the
        handle, so a caller that has spilled a large payload to a temp file never
        has to materialize it in memory.
        """
        await self.put(key, fileobj.read())

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def presigned_url(self, key: str, expires: int = 86400) -> str: ...

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
