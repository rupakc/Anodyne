# Anodyne — Generation C0 (Foundation + Tabular-from-Description Slice) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C0
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) · Foundation + LLM (merged)

## Goal

The Generation Engine foundation, proven by a complete vertical slice: a signed-in user describes a
tabular dataset in natural language, reviews the LLM-proposed schema, generates it (durably
orchestrated by Temporal, executed on Ray), and downloads a Parquet artifact — **all runnable
locally via `make up`**, through the Web UI. This spec introduces Temporal, Ray, the dataset domain
model, the `Generator` port, and the Web UI shell that C1–C6 build on.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Tabular rows (C0) | Deterministic, seeded Faker/distribution sampler from the schema; LLM used only for schema-from-description |
| Ray | Ray **head service in docker-compose**; workers connect to it (`ray.init(address=...)`) |
| Temporal | Temporal auto-setup service in docker-compose; durable `GenerationWorkflow` |
| Web UI | Full Keycloak-OIDC login (Auth.js) + create-from-description wizard + live progress + download; autumn-pastel design system |
| Offline LLM | Ollama service bundled in compose (offline default) **and** external model registration (user supplies key) |

## New packages / apps

- `anodyne-dataset` — domain models + ports (no infra imports).
- `anodyne-generation` — `LLMSchemaProposer`, `TabularSampler`.
- `anodyne-compute` — Ray connection + shard-task helpers.
- `anodyne-workflows` — Temporal `GenerationWorkflow` + activities.
- `apps/generation-worker` — Temporal worker process dispatching Ray work.
- `apps/web` — Next.js + TypeScript + Tailwind/shadcn UI.
- Extensions to `anodyne-storage` (tables + repository) and `apps/api-gateway` (endpoints).

## Components

### 1. Domain model — `anodyne-dataset`
- `Modality` — enum (`tabular` used in C0; others reserved).
- `SemanticType` — enum (e.g. `integer`, `float`, `categorical`, `boolean`, `datetime`, `name`,
  `email`, `address`, `text`).
- `FieldSpec` — name, semantic_type, nullable, constraints (min/max/choices/regex/…),
  distribution hint (e.g. `uniform`, `normal(mu,sigma)`, `categorical(weights)`).
- `DatasetSpec` — id, tenant_id, name, description, modality, `source` (`description` in C0),
  `schema: list[FieldSpec]`, `target_rows`, `directives` (reserved), status, created_at.
- `GenerationJob` — id, tenant_id, dataset_id, status (`pending`/`running`/`awaiting_review`/
  `succeeded`/`failed`), progress (0–1), message, workflow_id.
- `DatasetVersion` — id, dataset_id, tenant_id, artifact_uri, format (`parquet`), row_count,
  checksum, created_at.
- Ports: `DatasetRepository`, `Generator` (`generate(spec, shard) -> ShardArtifact`),
  `SchemaProposer` (`propose(description) -> list[FieldSpec]`).

### 2. Generation logic — `anodyne-generation`
- `LLMSchemaProposer(SchemaProposer)` — builds a structured prompt, calls `anodyne-llm`
  `LLMProvider.complete`, parses a JSON schema into `list[FieldSpec]` (validated with Pydantic;
  clear error on malformed output). Works with any registered model incl. Ollama.
- `TabularSampler(Generator)` — given a `DatasetSpec` schema, a row range, and a **seed**, produces a
  deterministic `pyarrow.Table` using Faker + numpy per `SemanticType`/distribution/constraints.
  Same seed + range ⇒ identical output (unit-tested).

### 3. Orchestration — `anodyne-workflows` + `anodyne-compute`
- `anodyne-compute`: `ray_task_generate(spec, shard, seed)` wrapping `TabularSampler` as a Ray remote;
  connection helper reading the Ray address from settings.
- `anodyne-workflows`: `GenerationWorkflow(job_id)` activities —
  `plan_shards` → (HITL schema-review gate: workflow waits for an approval signal) →
  `generate_shards` (fan out Ray tasks) → `assemble_parquet` → `upload_artifact` →
  `register_version` → `complete`. Progress published to Redis (→ gateway WebSocket).
  Retries/timeouts on activities; workflow is resumable.
- `apps/generation-worker`: hosts the Temporal worker, registers the workflow + activities,
  connects to Ray.

### 4. Storage — `anodyne-storage`
New tenant-scoped tables `datasets`, `dataset_versions`, `generation_jobs` (each `tenant_id` + RLS
policy) via a new Alembic migration; `SqlDatasetRepository` implementing `DatasetRepository` through
`tenant_session`.

### 5. Gateway API — `apps/api-gateway`
- `POST /datasets` — create a `DatasetSpec` from a description; synchronously calls
  `LLMSchemaProposer` and returns the proposed schema for review. (`datasets:write`)
- `GET /datasets/{id}`, `GET /datasets`, `PATCH /datasets/{id}` — review/edit the schema. (`datasets:read`/`write`)
- `POST /datasets/{id}/generate` — start the Temporal `GenerationWorkflow`; returns a `GenerationJob`. (`datasets:write`)
- `GET /jobs/{id}` — job status/progress; `WS /jobs/{id}/stream` — live progress from Redis. (`datasets:read`)
- `GET /datasets/{id}/versions`, `GET /datasets/{id}/versions/{v}/download` — presigned URL. (`datasets:read`)
- New RBAC permissions `datasets:read`/`datasets:write` added to the role map.

### 6. Web UI — `apps/web` (Next.js + TypeScript + Tailwind/shadcn, autumn-pastel)
- Keycloak OIDC login via Auth.js; authenticated API calls carry the access token.
- **Create-from-description wizard:** describe → review/edit proposed schema → set row count →
  generate.
- **Progress view:** live job progress via the WebSocket stream.
- **Dataset browser:** list datasets/versions; download artifact.
- Autumn-pastel design system (soft ambers/terracotta/dusty rose/sage/cream) encoded in the Tailwind
  theme; built with the `frontend-design` skill.

### 7. Infra / local run — `infra/docker`
`docker-compose` adds **Temporal (auto-setup)**, **Ray head**, **Ollama** to the existing
Postgres/Redis/MinIO/Keycloak. `make up` starts the full backbone; `make dev` runs `api-gateway`,
`generation-worker`, and `web`. Documented offline flow (Ollama, no keys) and external-model flow
(user supplies an API key at runtime). `.env.example` extended (Temporal/Ray/Ollama addresses).

## Testing strategy (TDD, no shortcuts)

- **Unit:** dataset models/ports; `TabularSampler` determinism + type/constraint honoring;
  `LLMSchemaProposer` with a mocked `LLMProvider` (valid + malformed output); `GenerationWorkflow`
  via Temporal's time-skipping test environment with mocked activities; each activity unit-tested.
- **Integration (Docker):** `SqlDatasetRepository` RLS isolation; a Ray-local shard generation; a
  full workflow run against the Temporal test server + MinIO producing a real Parquet artifact.
- **Web:** type-check + lint; a Playwright happy-path e2e (login → describe → generate → download),
  marked `e2e`, runnable against the local stack.

## Definition of done

`make up` + `make dev` locally: a demo user logs into the Web UI, describes a tabular dataset,
reviews the LLM-proposed schema (via Ollama offline or a registered model), generates it (Temporal
workflow + Ray), watches progress, and downloads a valid Parquet file. Unit suite green; integration
+ e2e green where Docker is available; `ruff`/`mypy --strict` clean; CI extended to cover the new
packages.

## Non-goals (C0)

Statistical/from-sample fidelity (C1); text/image/audio/video (C2–C5); template catalog +
bias/edge-case directives (C6); GPU model serving; production Ray/Temporal scaling and autoscaling.
