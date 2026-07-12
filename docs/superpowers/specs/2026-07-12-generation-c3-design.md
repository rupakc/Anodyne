# Anodyne — Generation C3 (Image Generation) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C3
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) · [Generation C0](./2026-07-12-generation-c0-design.md) (merged, main)

## Goal

Image dataset generation, provider-agnostic per the architecture decision: a tenant describes an
image dataset (+ optional labels/directives), the system derives per-item prompts, generates images
via a pluggable `ImageProvider` (self-hosted GPU OSS model, or an external API), and stores the
image files + a JSON manifest (prompt/label per item) in object storage — wired through the same
Temporal + Ray flow C0 established for tabular, dispatched by `DatasetSpec.modality`.

**Environment constraint:** no GPU or provider API keys are available to build/test this spec. The
full `ImageProvider` abstraction + one external-API adapter (OpenAI Images) + one self-hosted-adapter
interface (SDXL-shaped, GPU pipeline injected) are built and wired, but every test mocks
`ImageProvider` (or injects a fake/local-Ray-only pipeline) — no live HTTP calls, no GPU inference.
Live runs need a GPU node (self-hosted) and/or a tenant-registered provider API key (external).

## Decisions

| Decision | Choice |
|---|---|
| `ImageProvider` port location | New package `anodyne-image` (not `anodyne-dataset`) — mirrors how `anodyne-llm` owns `LLMProvider`'s concrete adapters while the generic `Generator`/`DatasetRepository` ports stay in `anodyne-dataset`. `anodyne-image` imports `Generator`, `DatasetSpec`, `FieldSpec`, `SemanticType` from `anodyne-dataset`. |
| Provider binding | One `ImageProvider` instance is constructed already bound to one tenant's model/key/base-url (mirrors `LLMSchemaProposer(provider, model_config)`, not `LLMProvider.complete(config, request)`'s per-call config) — `generate(prompt, *, seed, size)` only. Simpler to test, and images don't need per-call provider switching the way schema-proposal-vs-invoke do for LLMs. |
| Per-tenant provider config storage | **New** table `image_provider_configs` (mirrors `model_configs` exactly: id/tenant_id/name/provider/model/params/secret_ref/api_base/enabled), reusing `anodyne_core.models.ModelConfig` as the value type — no new Pydantic model. Kept **separate** from `model_configs` so an image-provider registration can never accidentally become "the tenant's LLM" picked by `get_schema_proposer`'s `configs[0]`. `SqlImageProviderRegistry` (in `anodyne-image`, mirroring `anodyne_llm.registry.SqlModelRegistry`) + `FernetSecretStore` reuse. |
| External adapter | `OpenAIImageProvider` — raw `httpx` POST to `{api_base or api.openai.com/v1}/images/generations` with `response_format=b64_json` (verified current shape via context7/openai-python: `POST /images/generations` → `{"data":[{"b64_json": "..."}]}`). Duck-typed HTTP client (any object with `async post(url, json=, headers=, timeout=)`) so tests inject a fake — no `httpx` mock-transport dependency needed. Replicate/fal.ai are structurally identical (POST + poll or POST + b64/url) and are follow-ups behind the same port; not built now to keep one adapter rigorous rather than three shallow ones. |
| Self-hosted adapter | `SelfHostedSDXLProvider` wraps an injected `DiffusionPipeline` callable (`(prompt, *, seed, size) -> bytes`) — the live shape of a GPU-resident `diffusers` SDXL pipeline. `RayGpuActorPipeline` adapts a Ray actor handle (`actor.generate.remote(...)`) into that callable, so the "served via Ray GPU actors" architecture decision has a concrete, testable (local-Ray, fake actor) wiring path without requiring `torch`/`diffusers`/a GPU as dependencies. Calling `generate()` with no pipeline configured raises a clear `ImageProviderError` (not a GPU crash). |
| Prompt derivation | `ImagePromptBuilder.build(spec, start_row, count) -> list[ImagePromptItem]` — deterministic, no RNG: item label rotates through `spec.fields`' first `CATEGORICAL` field's `choices` (or `spec.directives["labels"]`) by `item_index % len(choices)`; prompt text composes `spec.description` + `label` + `spec.directives` (`bias`/`use_case`/`edge_case`/`style` keys, C6's directive vocabulary) into one string. Same spec+range ⇒ identical prompts (the "deterministic-in-shape" requirement); only the provider call varies (and is mocked in tests). |
| `Generator` conformance | `ImageGenerator(Generator)` implements the existing `generate(spec, start_row, count, seed) -> pyarrow.Table` port signature exactly (worker still "selects a `Generator` by `spec.modality`"). Returns columns `item_index, label, prompt, image_bytes (binary), mime_type` — the per-item payload, Parquet-shard-shaped like `TabularSampler`'s output so it flows through the *same* shard-bytes → object-store → assemble two-stage pipeline C0 built. `ImageProvider.generate` is async; `ImageGenerator.generate` is sync (port contract) and drives it via `asyncio.run(...)` — safe because it always runs inside a plain (non-async) Ray remote task process, never inside an already-running event loop. |
| Modality dispatch — minimal + scoped | **Workflow orchestration (`GenerationWorkflow.run`) is untouched** — it already just names activities generically. `GenerationInput` gains one additive field, `modality: str = "tabular"` (default preserves every existing caller). The three activities that *are* modality-specific (`generate_shards`, `assemble_and_upload`, `register_version`) each gain a small `if` branch at the top delegating to a **new, separate module** `anodyne_workflows.image_activities` when `modality == "image"`; the existing tabular code path is untouched below the branch. This is the deliberately small, clearly-scoped touch to the shared C0 file the brief anticipated other modality specs (C2/C4/C5) will also make — a controller reconciling branches serially only has to interleave independent `if` blocks, not resolve overlapping rewrites. |
| Manifest assembly | `generate_shards` (image branch) Ray-generates each shard as a Parquet blob (same as tabular) containing the raw image bytes; `assemble_and_upload` (image branch) reads every shard, writes each row's `image_bytes` out as its own object (`datasets/{id}/{job}/images/{item_index}.png`), and uploads one `manifest.json` (`{"items":[{item_index,label,prompt,object_key,mime_type}, ...]}`) — the final artifact. `register_version` records `format="image_manifest"` (vs `"parquet"`) so downstream consumers (Export, Evaluation) can tell manifests from tabular artifacts. |
| Gateway routes | **Additive, not a rewrite of `create_dataset`:** new `POST /datasets/image` builds an `IMAGE`-modality `DatasetSpec` directly (no LLM schema proposal — images don't need a column schema; an optional `labels: list[str]` becomes one synthesized `CATEGORICAL` field). `start_generation`'s "no fields" guard is qualified to tabular only (`if spec.modality is Modality.TABULAR and not spec.fields`) since image datasets may have zero labels (single-class), and it now passes `modality=str(spec.modality)` into `GenerationInput`. New `POST/GET/DELETE /image-providers` CRUD (mirrors `/models`) over the new registry, gated by new `image_providers:read/write/delete` permissions. |
| Testing | Every `ImageProvider` in every test is a fake/mock (no network, no GPU). Ray-touching tests (image shard generation, the Ray-actor-pipeline wiring) use **local-mode Ray**, marked `integration`, mirroring `packages/anodyne-compute/tests/test_ray_tasks.py` exactly — no live cluster, no GPU. Postgres-backed registry tests use `testcontainers`, marked `integration`, mirroring `test_dataset_repo.py`. |

## New package: `anodyne-image`

- `models.py` — `ImagePromptItem` (item_index, label, prompt), `GeneratedImage` (data: bytes,
  mime_type), `ImageManifestEntry`.
- `ports.py` — `ImageProvider(ABC)`.
- `errors.py` — `ImageProviderError`.
- `prompts.py` — `ImagePromptBuilder`.
- `providers/openai.py` — `OpenAIImageProvider` (external API).
- `providers/selfhosted.py` — `SelfHostedSDXLProvider` + `DiffusionPipeline` protocol +
  `RayGpuActorPipeline`.
- `factory.py` — `resolve_image_provider(config, api_key) -> ImageProvider`, `register_provider`
  (small registry keyed on `ModelConfig.provider`; extensible for Replicate/fal.ai later).
- `generator.py` — `ImageGenerator(Generator)`.
- `registry.py` — `SqlImageProviderRegistry` (per-tenant `ModelConfig` CRUD over
  `image_provider_configs`, reusing `anodyne_storage.db.tenant_session` + `SecretStore`).

## Touched (additive) files

- `anodyne-storage`: new table `image_provider_configs` in `db.py` + `_TENANT_TABLES`; migration
  `0003_image_provider_configs.py`.
- `anodyne-compute`: new `image_tasks.py` (`generate_image_shard_bytes`,
  `remote_generate_image_shard`) — `ray_tasks.py` untouched.
- `anodyne-workflows`: new `image_activities.py`; `workflow.py` gains one dataclass field;
  `activities.py` gains one small `if`-branch each in `generate_shards`/`assemble_and_upload`/
  `register_version` + two new optional `ActivityContext` fields (`image_registry`,
  `secret_store`) defaulting to `None`.
- `apps/generation-worker`: `WorkerDeps`/`build_worker`/`main` wire the new registry + secret
  store (all additive, default `None`); `config.py` gains `secret_key`.
- `apps/api-gateway`: new routes (above); `authz.py` gains `image_providers:*` permissions.

## Non-goals (C3)

Live GPU inference or live provider API calls (no GPU/keys in this environment — documented, not
built as a runnable path here); Replicate/fal.ai adapters (same port, straightforward follow-ups);
image quality/dedup filtering (mirrors C2's text quality-filter concern, deferred); template catalog
+ full bias/edge-case directive vocabulary (C6 — C3 only *consumes* `spec.directives` generically).
