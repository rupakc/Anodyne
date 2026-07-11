# Anodyne — Architecture

> Living architecture document. Kept in sync with the codebase. Individual sub-systems
> are designed in dated specs under `docs/superpowers/specs/` and recorded as ADRs under `docs/adr/`.

Anodyne is a synthetic data generation and benchmarking platform. It generates
multimodal datasets from descriptions, templates, or sample data; injects controlled
noise/drift/outliers/bias; exports to multiple formats; and benchmarks the results with an
LLM-as-a-Judge mixture-of-experts evaluation pipeline — all multi-tenant, cloud-agnostic,
and deployable on GCP or on-prem.

## 1. Guiding principles

- **Cloud-agnostic by construction.** Application code depends only on portable primitives:
  containers, Postgres, the S3 API, and OIDC. Every cloud-specific detail lives in IaC
  (Terraform) and a thin platform-adapter layer. On-prem is *the same containers* on any Kubernetes.
- **Ports & adapters (hexagonal) in every service.** Domain logic depends on interfaces (ports);
  LLM providers, object storage, database, identity, and secrets are swappable adapters.
  This is how we uphold SOLID and separation of concerns and stay LLM-agnostic.
- **Async and job-oriented.** Temporal owns *flow* (durable, resumable, human-in-the-loop-pausable
  workflows); Ray owns *compute* (distributed generation, perturbation, and evaluation).
- **Multi-tenant everywhere.** `tenant_id` on every row, Postgres row-level security (RLS) for data
  isolation, Keycloak realms/groups for identity, and object-storage prefix isolation for artifacts.
- **DRY / YAGNI / KISS.** Shared logic lives in `packages/`; we add distribution, warehouses, and
  finer service splits only when a real workload demands them.

## 2. Technology spine

| Concern | Choice | Rationale |
|---|---|---|
| Backend services | Python 3.12+, FastAPI (async), Pydantic v2 | The data/ML ecosystem (PyArrow, Polars, SDV, HuggingFace, LLM SDKs) is Python-native |
| Distributed compute | Ray + Ray Data | One framework scales generation, perturbation, and multi-agent evaluation; runs local, GKE, on-prem |
| Workflow orchestration | Temporal | Durable, resumable workflows survive restarts and pause for HITL approval; self-hostable |
| Identity & multi-tenancy | Keycloak (OIDC) + Postgres RLS | Open-source, self-hostable, realm-per-tenant; RLS enforces isolation at the data layer |
| Metadata store | PostgreSQL | Relational metadata + RLS |
| Object storage | S3-compatible (GCS in cloud, MinIO on-prem) | Identical API across environments |
| Cache / ephemeral / pub-sub | Redis | Caching, rate limiting, job coordination, live-progress pub/sub |
| LLM abstraction | LiteLLM behind a thin Anodyne interface | Normalizes 100+ cloud providers + local (Ollama, vLLM); we wrap it to avoid lock-in |
| Frontend | Next.js (React) + TypeScript + Tailwind/shadcn | SSR/streaming for large reports; strong dashboard/wizard/annotation ergonomics |

## 3. Service topology & deployment

**Deployable units**

- `web` — Next.js frontend.
- `api-gateway` — FastAPI: OIDC token validation, tenant resolution, RBAC, rate limiting,
  REST + WebSocket (live job progress). The only public entry point.
- `generation-worker` — Ray-backed Temporal activity worker for generation, perturbation, export.
- `evaluation-worker` — Ray-backed Temporal activity worker for the LLM-as-a-Judge MoE pipeline.

**Backbone services:** Temporal, Postgres, Redis, object store (MinIO/GCS), Keycloak, LiteLLM.

**GCP deployment mapping**

- **Cloud Run** — stateless `web`, `api-gateway`, and light workers (scale-to-zero, spiky traffic).
- **GKE** — the full platform including the Ray cluster, Temporal, and GPU node pools for local
  model inference and image/audio/video generation.
- **Cloud Functions** — thin event hooks / webhook receivers only.
- **On-prem** — identical containers via Helm on any Kubernetes, with MinIO + self-hosted
  Keycloak/Temporal.

## 4. Data & control flow

```
web → api-gateway (authn/z + tenant)
    → create Job record (Postgres)
    → start Temporal workflow
        → Ray activities: generate → perturb → export  (artifacts → object store)
        → HITL gate: workflow blocks on approval signal (may wait indefinitely)
        → evaluation workflow: mixture-of-experts LLM judges
        → report artifact (object store)
    → user reviews / annotates → feedback re-enters the loop
```

Metadata lives in Postgres. Large datasets, media, and reports live in the object store, referenced
by URI. Live progress flows Redis pub/sub → WebSocket → UI.

## 5. Cross-cutting concerns

- **Observability.** OpenTelemetry in all services → OTel Collector → Prometheus (metrics) +
  Tempo/Jaeger (traces) + Loki (logs) + Grafana. Fully portable.
- **Security.** OIDC; per-tenant encryption of bring-your-own model keys via a secrets abstraction
  over Vault / cloud KMS; RLS; audit logging. CI enforces SBOM, image signing, dependency and secret scanning.
- **IaC / CI-CD.** Terraform modules (GCP + generic K8s) + Helm charts. GitHub Actions:
  lint / type-check / test → build → scan → sign → push → deploy per environment.
- **Configuration.** 12-factor, env-driven via Pydantic Settings; feature flags.

## 6. Monorepo layout

```
anodyne/
  apps/
    web/                 # Next.js frontend
    api-gateway/         # FastAPI: auth, tenancy, RBAC, routing, WebSocket
    generation-worker/   # Ray + Temporal activities: generate/perturb/export
    evaluation-worker/   # Ray + Temporal activities: MoE LLM-as-a-Judge
  packages/              # Python libraries (uv workspace)
    anodyne-core/        # domain models, ports/interfaces, shared types (Pydantic)
    anodyne-llm/         # LLM abstraction (LiteLLM adapter)
    anodyne-generation/  # templates, from-sample, multimodal generators
    anodyne-perturbation/# noise, drift, outliers, bias/edge-case injection
    anodyne-export/      # CSV/JSON/Parquet/Arrow, size-based format defaults
    anodyne-evaluation/  # MoE judges, scoring, report generation
    anodyne-storage/     # object-store + database adapters
    anodyne-tenancy/     # auth, RLS, tenant context
    anodyne-workflows/   # Temporal workflow/activity definitions
    anodyne-observability/# OpenTelemetry setup, logging
  infra/
    terraform/           # GCP + generic K8s modules
    helm/                # charts for all services + backbone
    docker/              # base images, compose for local dev
  docs/                  # architecture.md, specs/, adr/
  .github/workflows/     # CI/CD
  .claude/               # CLAUDE.md, agents/, skills/, hooks/
```

Python is managed with a **uv workspace**; `web` uses **pnpm + Turborepo**.

## 7. Sub-system build roadmap

Each sub-system gets its own spec → plan → implementation cycle.

| Order | Sub-system | Requirements |
|---|---|---|
| A (now) | Platform Foundation — monorepo, multi-tenant identity + RBAC, gateway, domain models, observability | 11, 14, 15 |
| B (now) | LLM Abstraction Layer — provider-agnostic interface, cloud + local | 10 |
| C | Generation Engine — description→schema, templates, from-sample, multimodal | 1, 2, 6 |
| D | Perturbation Module — noise, drift, outliers, bias/edge-case | 3, 4 |
| E | Export & Storage — formats + size-based defaults | 5 |
| F | Evaluation Engine — LLM-as-a-Judge MoE, reports | 7, 8, 9 |
| G | Human-in-the-loop & Annotation — review, feedback, annotation | 12, 13 |
| H | Web UI — built incrementally alongside C–G | presentation |
| I | Deployment & CI/CD — GCP + on-prem, GitHub Actions | infra |
