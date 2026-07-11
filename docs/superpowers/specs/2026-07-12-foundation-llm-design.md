# Anodyne — Foundation + LLM Abstraction (Walking Skeleton) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-systems A (Platform Foundation) + B (LLM Abstraction Layer)
- **Depends on:** [Top-level architecture](./2026-07-12-anodyne-architecture-design.md) · [`docs/architecture.md`](../../architecture.md)

## Goal

Build the shared platform spine as a **thin vertical walking skeleton**: the smallest slice that
exercises identity, multi-tenancy, storage, observability, and the LLM abstraction end-to-end.
Success = a health check plus one authenticated, tenant-scoped, RBAC-checked `POST /llm/invoke`
that calls a per-tenant registered model (cloud or local) through the LLM port. No generation,
perturbation, export, evaluation, or Temporal yet.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| First-spec scope | Thin vertical walking skeleton |
| Tenant model | Single Keycloak realm + Organizations/groups; `tenant_id` claim in token |
| RBAC | Fixed roles (owner/admin/member/viewer) behind an `AuthorizationPolicy` port |
| LLM layer | Embedded LiteLLM SDK behind an `LLMProvider` port; per-tenant model configs with encrypted keys in Postgres |

Library versions are pinned during planning/implementation against current docs (context7); this
spec fixes architecture and behavior, not versions.

## Packages in scope

`anodyne-core`, `anodyne-tenancy`, `anodyne-storage`, `anodyne-observability`, `anodyne-llm`,
and the `apps/api-gateway` app. All other `packages/` are created empty or deferred to their
roadmap stage.

## Components

### 1. Domain models & ports — `anodyne-core`

Pydantic v2 models (no infrastructure imports):

- `Tenant` — id, name, org_ref, status, created_at
- `User` — id, tenant_id, subject (OIDC `sub`), email, roles
- `Role` — enum: `owner`, `admin`, `member`, `viewer`
- `ModelConfig` — id, tenant_id, name, provider, model, params, `secret_ref`, optional `api_base`
  (local models), enabled
- Value objects — `TenantContext` (tenant_id, user, roles), `LLMRequest` (messages, model_config_id,
  params), `LLMResponse` (content, usage, cost, latency)

Ports (ABC/Protocol), depended on by domain logic, implemented by adapters elsewhere:

- `ObjectStore` — put / get / presigned_url / list
- `SecretStore` — encrypt / decrypt / store / resolve
- `LLMProvider` — complete / stream
- `AuthorizationPolicy` — `is_permitted(context, permission)`
- `UnitOfWork` — transactional DB session boundary

### 2. Identity & tenancy — `anodyne-tenancy`

- **OIDC validation.** Verify Keycloak-issued JWTs with cached JWKS; extract `sub`, email, tenant
  (Organization) claim, and realm roles. `get_tenant_context()` FastAPI dependency builds
  `TenantContext`; requests without a resolvable tenant are rejected (401/403).
- **Postgres RLS.** Every tenant-scoped table carries `tenant_id` and an RLS policy. Each
  transaction issues `SET LOCAL app.tenant_id = :tid`; an RLS-aware async session factory binds the
  current `TenantContext` so isolation is enforced in the database, not just application code.
- **RBAC.** `RoleBasedPolicy` implements `AuthorizationPolicy`, mapping roles → permitted actions.
  A `require(permission)` dependency guards endpoints. Swappable for a policy engine later.

### 3. Storage — `anodyne-storage`

- **Object store.** S3 adapter (boto3) implementing `ObjectStore`, keys tenant-prefixed
  `{tenant_id}/…`; works against MinIO (on-prem/dev) and GCS interop (cloud).
- **Database.** SQLAlchemy 2.0 async + asyncpg; Alembic migrations; RLS-aware session/`UnitOfWork`.
- **Secrets.** `SecretStore` with envelope encryption. Dev adapter uses a Fernet symmetric key;
  prod adapter targets Vault / cloud KMS. Only encrypted `secret_ref`s are persisted in Postgres.

### 4. Observability — `anodyne-observability`

OpenTelemetry traces/metrics/logs initialization; `structlog` JSON logging correlated by
`tenant_id` and `request_id`; FastAPI middleware that opens a span and binds log context per request.

### 5. LLM layer — `anodyne-llm`

- `LLMProvider` port: `complete(LLMRequest) -> LLMResponse`, `stream(LLMRequest) -> AsyncIterator`.
- `LiteLLMAdapter`: resolves a `ModelConfig` → LiteLLM call (API key fetched via `SecretStore`,
  `api_base` set for Ollama/vLLM), invokes `litellm.acompletion`, normalizes content + usage + cost.
- `ModelRegistry`: per-tenant CRUD over `ModelConfig` plus a `test_connection()` probe call.
- Cloud (OpenAI/Anthropic/…) and local (Ollama/vLLM) models are handled through one code path.

### 6. API gateway — `apps/api-gateway`

Endpoints:

- `GET /healthz`, `GET /readyz` — liveness/readiness (no auth).
- `GET /me` — current `TenantContext`.
- `POST /models`, `GET /models`, `DELETE /models/{id}` — register/list/remove tenant model configs;
  keys encrypted on write via `SecretStore`.
- `POST /llm/invoke` — **the skeleton proof**: authenticated, tenant-scoped, RBAC-checked; invokes a
  registered model through `anodyne-llm` and returns completion + usage.

Middleware order: auth → tenant resolution → tracing/logging → error handling.

### 7. Local dev & tooling

- `infra/docker/docker-compose.yml`: Postgres, Redis, Keycloak (preseeded realm + Organization +
  demo users via realm import), MinIO.
- `uv` workspace; `ruff` (lint + format), `mypy` (strict), `pytest` + `pytest-asyncio`, coverage;
  `pre-commit` hooks.
- `.claude/CLAUDE.md`: dev conventions and commands.
- GitHub Actions **quality gate** (lint / type-check / test on PR). Full IaC/CD is stage I.

## Testing strategy (TDD)

- **Unit:** `RoleBasedPolicy` decisions; tenant key-prefixing; secret encrypt/decrypt round-trip;
  `LiteLLMAdapter` with LiteLLM mocked; OIDC claim extraction.
- **Integration:** RLS isolation (tenant A cannot read tenant B's rows); OIDC token validation
  against test JWKS; object store against MinIO; `POST /llm/invoke` against a mock/local model.

## Non-goals

Generation, perturbation, export, evaluation, Temporal workflows, the full Web UI, Terraform/Helm
IaC and full CD, and GPU inference. Each is delivered in its own roadmap stage.

## Definition of done

`docker-compose up` brings up the backbone; migrations + realm seed apply; a demo user in a tenant
can authenticate, register a model, and receive a completion from `POST /llm/invoke`; RLS isolation
and the listed tests pass; the CI quality gate is green.
