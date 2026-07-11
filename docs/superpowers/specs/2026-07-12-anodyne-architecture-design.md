# Anodyne — Top-Level Architecture Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Canonical living doc:** [`docs/architecture.md`](../../architecture.md)

This is the accepted design record for Anodyne's top-level architecture. It captures the
decisions and their rationale at a point in time; `docs/architecture.md` is the living document
kept in sync with the code.

## Problem

Build Anodyne: a multi-tenant, cloud-agnostic platform for synthetic data generation
(multimodal, from description/template/sample, with controlled noise/drift/outlier/bias
injection and multi-format export) and LLM-as-a-Judge mixture-of-experts benchmarking, with
human-in-the-loop review and annotation, deployable on GCP (Cloud Run / GKE / Cloud Functions)
and on-prem, with a secure GitHub CI/CD pipeline.

The request spans nine independent sub-systems. It is decomposed into per-sub-system
spec→plan→implementation cycles (roadmap A–I in `architecture.md` §7). This spec covers only
the top-level architecture that all sub-systems share.

## Decisions

| Decision | Choice | Alternatives rejected |
|---|---|---|
| Backend stack | Python 3.12+, FastAPI, Pydantic v2 | Polyglot; Node/TS everywhere (cuts off Python ML ecosystem) |
| Distributed compute | Ray + Ray Data | Dask+Celery, Spark (weaker multimodal/LLM story), defer-distribution |
| Orchestration | Temporal (durable workflows) | Ray+Postgres state machine, task queue (weak HITL durability) |
| Identity / multi-tenancy | Keycloak (OIDC) + Postgres RLS | Managed OIDC (vendor lock), custom auth (high risk) |
| Storage backbone | Postgres + S3-compatible object store + Redis | +warehouse now (YAGNI), Postgres+objstore only (Redis needed soon) |
| LLM layer | LiteLLM behind a thin Anodyne interface | Custom per-provider adapters, direct SDKs (violate agnosticism) |
| Frontend | Next.js + TypeScript + Tailwind/shadcn | Vite SPA, Remix/TanStack Start |
| Topology | Modular monorepo, few deployable services | Microservices day 1 (premature), single monolith (bad gen/eval scaling) |

## Architecture summary

Four deployable units (`web`, `api-gateway`, `generation-worker`, `evaluation-worker`) over a
backbone of Temporal, Postgres, Redis, object store, Keycloak, and LiteLLM. Hexagonal
(ports & adapters) inside each service; shared domain and adapters in `packages/`. Temporal
orchestrates durable, HITL-pausable pipelines; Ray executes distributed compute. See
`architecture.md` §3–§6 for topology, data flow, cross-cutting concerns, and monorepo layout.

## Deployment

GCP: Cloud Run (stateless services), GKE (full platform + Ray + GPU node pools), Cloud Functions
(thin event hooks). On-prem: identical containers via Helm on any Kubernetes. Terraform + Helm
for IaC; GitHub Actions for secure CI/CD.

## Out of scope for this spec

Detailed designs of each sub-system (generation algorithms, perturbation methods, the MoE
evaluation pipeline, HITL/annotation UX, export internals, CI/CD stages). Each is its own spec.

## Next step

Brainstorm the first buildable sub-project: **Platform Foundation + LLM Abstraction Layer**
(roadmap A + B).
