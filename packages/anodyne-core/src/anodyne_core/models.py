from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class Role(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Tenant(BaseModel):
    id: UUID
    name: str
    org_ref: str
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class User(BaseModel):
    id: UUID
    tenant_id: UUID
    subject: str  # OIDC `sub`
    email: str
    roles: list[Role] = Field(default_factory=list)


class TenantContext(BaseModel):
    tenant_id: UUID
    user: User
    roles: list[Role]

    def has_role(self, role: Role) -> bool:
        return role in self.roles


class ModelConfig(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    provider: str  # e.g. "openai", "anthropic", "ollama", "vllm"
    model: str  # e.g. "gpt-4o"
    params: dict[str, object] = Field(default_factory=dict)
    secret_ref: str | None = None  # encrypted-secret handle; None for keyless local models
    api_base: str | None = None  # set for local models (Ollama/vLLM)
    enabled: bool = True


class Message(BaseModel):
    role: str
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    model_config_id: UUID
    messages: list[Message]
    params: dict[str, object] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    usage: Usage
    cost: float = 0.0
    latency_ms: float = 0.0
