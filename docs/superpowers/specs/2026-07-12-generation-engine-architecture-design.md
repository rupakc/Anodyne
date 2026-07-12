# Anodyne — Generation Engine Architecture Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C (Generation Engine), spanning requirements 1, 2, 4 (generation side), 6
- **Depends on:** [Top-level architecture](./2026-07-12-anodyne-architecture-design.md) · Foundation + LLM (merged)

This is the accepted architecture for Anodyne's Generation Engine. It covers all five modalities
coherently and is delivered as a sequence of rigorous per-capability implementation specs
(C0–C6), each built to the same TDD + review bar as Foundation. `docs/architecture.md` is the
living document.

## Goal

Generate multimodal synthetic datasets — tabular, text, image, audio, video — from three sources
(natural-language description, starter template, or an uploaded sample), steerable toward specific
biases / use-cases / edge-cases, orchestrated durably (Temporal) and executed at scale (Ray),
usable **locally** via `make up` and through a **Web UI**.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Modality scope | All five (tabular, text, image, audio, video), comprehensive, behind a modality-pluggable abstraction |
| Generation methods | Both families behind a `Generator` port: LLM-based (`anodyne-llm`) and statistical/ML |
| Tabular synthesis | Permissive stack (copulas/rdt [MIT], CTGAN/TVAE) as default; **SDV as an opt-in, separately-licensed adapter** (BSL 1.1) — both behind the port |
| Image/audio/video | Provider-agnostic: self-hosted OSS on GPU (SDXL, XTTS/Bark, text-to-video) via Ray + GPU node pools, AND external APIs (OpenAI Images, ElevenLabs, Replicate, fal.ai, Runway) — behind per-modality provider ports |
| Orchestration | Temporal (durable, HITL-pausable workflows) — introduced now |
| Compute | Ray + Ray Data (distributed generation + GPU inference) — introduced now |
| Local run | Full stack (backbone + Temporal + Ray head + gateway + generation-worker + web) via docker-compose / `make up` |
| UI | Next.js `apps/web`, built incrementally with each spec, autumn-pastel design system |

## 1. Dataset domain model — new package `anodyne-dataset`

The shared spine consumed by Generation (C), Perturbation (D), Export (E), Evaluation (F).

- `Modality` — enum: `tabular`, `text`, `image`, `audio`, `video`.
- `FieldSpec` — name, semantic type, constraints, optional distribution hints (tabular/text schema).
- `DatasetSpec` — id, tenant_id, name, modality, `source` (description | template | sample),
  target schema, target size, `directives` (bias/use-case/edge-case steering), status.
- `Profile` — statistics inferred from an uploaded sample (schema, per-field distributions,
  correlations for tabular; corpus/style stats for other modalities).
- `DatasetVersion` — immutable artifact reference (object-store URI + format + row/item count +
  checksum) with lineage back to the `DatasetSpec` and generation `Job`.
- Ports: `DatasetRepository` (metadata CRUD, RLS), `Generator`, `SampleProfiler`.

Metadata → Postgres (tenant-scoped, RLS). Data/media → object store: tabular as Parquet/Arrow;
media as files + a JSON manifest describing items and their labels/prompts.

## 2. Generation sources & the Generator port hierarchy — `anodyne-generation`

- **Sources:** from-description (LLM proposes a schema via `anodyne-llm`; user reviews/customizes),
  from-template (starter-template catalog → spec, customizable), from-sample (`SampleProfiler`
  infers schema + distributions → synthesis plan).
- **`Generator` port:** `generate(spec, profile, shard) -> ShardArtifact`. One `ModalityGenerator`
  implementation per modality, selected by `spec.modality`.
- **Media provider ports:** `ImageProvider`, `AudioProvider`, `VideoProvider`, each with
  self-hosted-OSS and external-API adapters; tenants register providers/models (reusing the
  `anodyne-llm` model-registry pattern and encrypted secrets).
- **`GenerationDirective`** (requirement 4): declarative steering on the spec — bias toward
  subpopulations, target named use-cases, or force edge-cases — applied via distribution
  constraints (tabular) or prompt conditioning (LLM/media).

## 3. Per-modality strategy

- **Tabular:** permissive statistical stack (`copulas`+`rdt`, CTGAN/TVAE) as default for from-sample
  fidelity; Faker/Mimesis for realistic field values; constraint enforcement; **SDV opt-in** adapter.
  Large volumes generated as Ray Data shards.
- **Text:** LLM-based via `anodyne-llm` — prompt templates, structured outputs, corpora for
  classification/QA/summarization/chat, with deduplication + quality filtering.
- **Image / Audio / Video:** provider-agnostic. Self-hosted OSS models served on Ray GPU actors
  (SDXL; XTTS/Bark; text-to-video), and external API adapters — behind the per-modality provider
  ports. Heavy artifacts streamed to the object store with a manifest.

## 4. Orchestration & execution — `anodyne-workflows` (Temporal) + `anodyne-compute` (Ray)

`gateway → create Dataset + Job → start Temporal GenerationWorkflow(job_id)` → activities:

1. **resolve/validate spec** (profile the sample if source = sample),
2. **plan shards** (split target size into work units),
3. **generate shards in parallel on Ray** (per-modality `Generator`; GPU for media),
4. **assemble + write artifact** to the object store,
5. **register `DatasetVersion`** and mark the job complete.

HITL-pausable schema-review gate between resolve and generate. Live progress via Redis → WebSocket.
New app `generation-worker` (a Temporal worker that dispatches Ray work). Backbone adds **Temporal**
and a **Ray head** to docker-compose and Helm; GPU node pools on GKE / on-prem for media.

## 5. API surface (gateway) & Web UI

- Gateway: `POST /datasets` (create spec) · `POST /datasets/{id}/sample` (upload) ·
  `POST /datasets/{id}/generate` · `GET /datasets/{id}` + `/versions` · `GET /jobs/{id}` (progress) ·
  `GET /datasets/{id}/versions/{v}/download` (presigned) · `GET /templates`.
- **Web UI (`apps/web`, Next.js + TypeScript, autumn-pastel):** OIDC login (Keycloak), a
  create-dataset wizard (description / template / sample), live generation progress, dataset/version
  browser, and artifact download. Built incrementally, one usable slice per spec.

## 6. Local runnability

`make up` brings the entire stack up locally: Postgres, Redis, MinIO, Keycloak, **Temporal**,
**Ray head**, plus `api-gateway`, `generation-worker`, and `web`. `make migrate`/`make seed` prepare
data. A documented dev flow lets a developer sign in and generate a dataset end-to-end on their
laptop (CPU-only path for tabular/text; media providers default to external APIs when no GPU).

## 7. Decomposition into implementation specs (each: spec → TDD plan → subagent execution → review)

| Spec | Scope |
|---|---|
| **C0** | Generation foundation + tabular vertical slice: `anodyne-dataset` model + artifact storage + `Generator` port + **Temporal & Ray wiring** + gateway endpoints + a **minimal Web UI**, proven end-to-end by generating a small tabular dataset *from a description* → artifact → download, locally via `make up`. |
| **C1** | Tabular (full): permissive synth stack + from-sample profiling + Faker + constraints + SDV opt-in + Ray-sharded scale. |
| **C2** | Text generation (LLM corpora, templates, structured output, quality filters). |
| **C3** | Image generation (provider port: self-hosted SDXL via Ray/GPU + external APIs). |
| **C4** | Audio generation (TTS + audio synthesis providers). |
| **C5** | Video generation (text-to-video providers). |
| **C6** | Template catalog + bias/edge-case/use-case directives. |

## New packages / apps introduced

`anodyne-dataset`, `anodyne-generation`, `anodyne-compute` (Ray), `anodyne-workflows` (Temporal),
app `generation-worker`, app `web` (Next.js). Backbone: Temporal + Ray head added to compose/Helm.

## Out of scope for this architecture

Perturbation (D), Export internals (E), Evaluation (F), and the full production GPU autoscaling /
deployment (I) — each its own sub-system. C0 is the next spec to brainstorm in detail.
