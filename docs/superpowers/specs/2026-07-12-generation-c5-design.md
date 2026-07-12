# Generation C5 — Video Generation Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C (Generation Engine), spec C5 of C0–C6
- **Depends on:** C0 (generation foundation: `anodyne-dataset`, `anodyne-workflows`,
  `anodyne-compute`, `api-gateway`/`generation-worker` wiring) — merged to `main`.

## Goal

Generate short synthetic video-clip datasets from a `DatasetSpec` (modality `video`),
provider-agnostic across self-hosted OSS text-to-video models (GPU, via Ray) and
external video-generation APIs (Runway / Replicate / fal.ai style). Store the clips as
files in the object store plus a JSON manifest describing each item; wire a
`modality="video"` path through the Temporal workflow and the generation worker.

**No GPU or provider API keys are available in this environment.** This spec builds the
full abstraction, one external-API adapter, and one self-hosted-adapter interface (+ a
reference implementation shape), but every test mocks the `VideoProvider` — no real
inference, no live network calls, no media binaries committed. Live runs need a GPU node
pool (self-hosted path) and/or a provider API key (external path); see "Operating live"
below.

## Why video doesn't reuse the tabular `Generator` port as-is

C0's `Generator` port (`anodyne_dataset.ports.Generator.generate(spec, start_row, count,
seed) -> pyarrow.Table`) is shaped for row-oriented, synchronous, CPU-cheap sampling —
right for tabular. Video generation is fundamentally different: each "row" is a
GPU-or-network-bound async call producing a large binary blob (a video file), not a
column value. Forcing video through the same synchronous, table-returning port would
mean either (a) smuggling raw video bytes through pyarrow binary columns (awkward,
memory-heavy, and loses the "files + manifest" storage shape the architecture doc calls
for), or (b) blocking Ray workers on network I/O with no natural batching. Instead:

- A new **`VideoProvider` port** (async) is the seam for "turn one prompt + params into
  one video asset," implemented by self-hosted and external adapters.
- A **`VideoDatasetGenerator`** (in `anodyne_video`) turns a `DatasetSpec` into N
  prompts and drives `VideoProvider` calls; it is the video analogue of `Generator` but
  async and manifest-shaped, not a `pyarrow.Table` producer.
- Dedicated Temporal activities (`anodyne_workflows.video_activities`) upload each clip
  to the object store and assemble a `manifest.json`; `DatasetVersion.format` becomes
  `"video-manifest"` for these jobs (vs. `"parquet"` for tabular).

This keeps the existing tabular pipeline (`activities.py`, its `ActivityContext`, the
five original activities) **completely untouched** — the video path is additive new
files plus a small modality branch in `GenerationWorkflow.run` and a few registration
lines in `generation-worker`. This bounds the blast radius for the controller
reconciling C1–C5 in parallel: each modality spec adds its own activities/context and
extends the same one `if/elif` in the workflow and the same registration list in the
worker, rather than rewriting shared tabular code.

`DatasetSpec.target_rows` is reused, for video jobs, as "target clip count" (documented,
not renamed — renaming a shared field is exactly the kind of cross-cutting change the
per-modality specs are meant to avoid; C6/a later cleanup can introduce a
modality-neutral `target_items` alias if useful).

## Domain model — new package `anodyne-video`

`packages/anodyne-video/src/anodyne_video/`:

- `models.py`:
  - `VideoProviderConfig` — tenant's registered provider (mirrors `ModelConfig`):
    `id, tenant_id, name, provider` (`"replicate"` | `"self-hosted"` | ...), `model`,
    `params`, `secret_ref` (encrypted, nullable for keyless self-hosted), `api_base`,
    `enabled`.
  - `VideoGenerationRequest` — `prompt, duration_seconds, width, height, fps, seed,
    params`.
  - `VideoAsset` — one generated clip: `content: bytes, content_type, duration_seconds,
    width, height, fps, seed, provider, model`.
  - `VideoManifestItem` — one manifest row (no bytes): `index, object_key, prompt,
    duration_seconds, width, height, fps, seed, provider, model, content_type,
    byte_size`.
  - `VideoManifest` — `tenant_id, dataset_id, job_id, items: list[VideoManifestItem],
    created_at`.
- `ports.py`:
  - `VideoProvider(ABC)` — `async def generate(self, config: VideoProviderConfig,
    request: VideoGenerationRequest) -> VideoAsset`.
  - `VideoProviderRegistry(ABC)` — `create/get/list/delete` tenant-scoped
    `VideoProviderConfig`s (mirrors `anodyne_llm.registry.SqlModelRegistry`'s shape,
    but declared as a formal port here since `anodyne-video` is a fresh package).
- `adapters/self_hosted.py`:
  - `SelfHostedVideoProvider(VideoProvider)` — wraps an injected synchronous "model
    runner" (`Callable[[VideoGenerationRequest], bytes]`), executed off the event loop
    via `asyncio.to_thread`. This is the seam a Ray-GPU-backed text-to-video model
    (e.g. a Stable-Video-Diffusion-family or ModelScope-T2V actor) plugs into: the
    runner passed at construction can be `lambda req: ray.get(actor.generate.remote(req))`
    against a Ray actor holding GPU model weights. Building that actor (real model
    weights + GPU scheduling) is explicitly out of scope here — no GPU is available in
    this environment — but the port and the dispatch shape are in place, and are
    exercised in tests with a fake runner.
- `adapters/external_api.py`:
  - `ReplicateVideoProvider(VideoProvider)` — external-API adapter over a generic
    "create prediction → poll → download" REST flow (the shape used by Replicate;
    fal.ai/Runway differ in wire format but fit the same three-step shape behind their
    own adapter — one more adapter is a follow-up, not a redesign). Takes an injected
    `httpx.AsyncClient` (tests use `httpx.MockTransport`, no network) and a
    `SecretStore` to decrypt `config.secret_ref` into a bearer token.
- `registry.py`:
  - `SqlVideoProviderRegistry(VideoProviderRegistry)` — SQL-backed, mirrors
    `SqlModelRegistry` exactly (tenant-scoped RLS session, encrypted secret_ref).
- `generator.py`:
  - `build_video_prompt(spec, index) -> str` — deterministic prompt from
    `spec.description` + `spec.directives` (e.g. `style`, `scene`, `subject` keys) + the
    item index (so shards are disjoint/reproducible, mirroring `TabularSampler`'s
    per-offset RNG seeding).
  - `VideoDatasetGenerator` — `async def generate_items(spec, provider, config,
    start_index, count, seed) -> list[tuple[VideoManifestItem, bytes]]`: builds one
    prompt per index, calls `provider.generate(config, request)`, and returns
    manifest-item + raw-bytes pairs (upload to the object store happens in the
    Temporal activity, not here — keeps this class storage-agnostic and unit-testable
    with a fake provider).

## Storage — additive changes to `anodyne-storage`

- `db.py`: new `video_provider_configs` table (same shape as `model_configs`), added to
  `_TENANT_TABLES` (RLS).
- `migrations/versions/0003_video_provider_configs.py`: creates the table + RLS policy,
  following `0002_datasets.py`'s pattern exactly.

No changes to `datasets`/`generation_jobs`/`dataset_versions` — `DatasetVersion.format`
already free-text (`"video-manifest"` is just a new value, no schema change).

## Temporal + worker wiring (additive)

- `anodyne_workflows/workflow.py`: `GenerationInput` gains `modality: str = "tabular"`
  (back-compat default — existing tabular call sites are unaffected). `run()` branches
  once on `inp.modality`:
  - `"tabular"` (default): unchanged five-activity sequence.
  - `"video"`: `plan_video_items` → `generate_video_items` → `assemble_video_manifest`
    → `register_video_version`, with the same `set_status` progress calls around it.
- `anodyne_workflows/video_activities.py` (**new file**, zero edits to the existing
  `activities.py`/`ActivityContext`):
  - `VideoActivityContext` (repo, s3 bucket/client, `SecretStore`, a `provider_registry:
    VideoProviderRegistry`, a `providers: dict[str, VideoProvider]` keyed by
    `VideoProviderConfig.provider`, optional publisher) + `configure_video_activities`.
  - `plan_video_items` — splits `target_rows` (item count) into shards of
    `_VIDEO_SHARD_ITEMS` (small, e.g. 4 — clips are heavy, unlike tabular's 50k-row
    shards).
  - `generate_video_items` — for each shard, resolves the tenant's enabled
    `VideoProviderConfig` (first enabled one; a later spec can add "pick by
    directive"), looks up the matching `VideoProvider` adapter, runs
    `VideoDatasetGenerator.generate_items`, uploads each clip to
    `datasets/{dataset}/{job}/videos/item-{index}.mp4`, and returns manifest-item dicts
    (byte content stripped before returning — activities must be JSON-serializable).
  - `assemble_video_manifest` — builds a `VideoManifest` from the items and uploads
    `datasets/{dataset}/{job}/manifest.json`.
  - `register_video_version` — writes a `DatasetVersion` with `format="video-manifest"`,
    `artifact_uri=<manifest key>`, `row_count=<item count>`.
- `apps/generation-worker/src/generation_worker/main.py`: imports + appends the four
  video activities to `registered_activities()`; `WorkerDeps`/`build_worker` gain
  optional `secret_store`, `video_registry`, `video_providers` fields (default `None` /
  empty) so the tabular-only worker configuration keeps working unchanged; `main()`
  wires a real `SqlVideoProviderRegistry` + a `ReplicateVideoProvider` when configured.
- `apps/generation-worker/tests/test_worker_wiring.py`: `EXPECTED_ACTIVITY_NAMES` grows
  to include the four new names (existing test, extended — this is the one
  pre-existing test file the video work must touch, since it asserts an exact set).

## Gateway (additive)

- `apps/api-gateway/src/api_gateway/app.py`: `start_generation` passes
  `modality=str(spec.modality)` to `GenerationInput` (one-line change — the field
  already existed on `DatasetSpec`).
- New routes mirroring `/models`: `POST /video-providers`, `GET /video-providers`,
  `DELETE /video-providers/{id}` — CRUD over `VideoProviderConfig`, secrets encrypted at
  rest via `SecretStore`, `secret_ref` never returned in responses (same pattern as
  `register_model`/`list_models`).
- `anodyne_tenancy/authz.py`: new permissions `video_providers:read/write/delete`,
  granted to the same roles as the equivalent `models:*` permissions (member:
  read+write, admin: +delete).
- `apps/api-gateway/src/api_gateway/deps.py`: `get_video_provider_registry` dependency
  (real `SqlVideoProviderRegistry`, overridable in tests).

## Multi-tenant / RLS / secrets

Same posture as `anodyne-llm`'s `ModelConfig`: `video_provider_configs` is tenant-scoped
with RLS (`tenant_session` + `_TENANT_TABLES`), API keys are Fernet-encrypted via the
existing `SecretStore` port before storage, and are never logged or returned to clients.

## Testing strategy (no GPU, no live network, no media binaries)

- Every `VideoProvider` implementation is exercised with fakes: the self-hosted adapter
  via an injected fake "model runner" callable; the external-API adapter via
  `httpx.MockTransport` (an in-process fake transport — no sockets, no `respx`
  dependency).
- `VideoDatasetGenerator`/prompt-building tests use a fake `VideoProvider`.
- Temporal activities are tested the same way `test_activities.py` tests the tabular
  ones: pure-function-style calls against `configure_video_activities` with fakes, plus
  a moto-mocked S3 bucket for the upload/manifest-assembly tests (never real network).
- `SqlVideoProviderRegistry` tests are `integration` (testcontainers Postgres), mirroring
  `anodyne_llm`'s `test_registry.py` exactly.
- Gateway route tests use `httpx.AsyncClient` against the FastAPI app with dependency
  overrides (in-memory fakes), mirroring `test_models_routes.py`.
- No test writes an actual video file, calls a real model, or calls a live provider API.
  Any generated "bytes" in tests are small fixed placeholders (e.g. `b"fake-mp4-bytes"`).

## Operating live (documented, not built here)

- **Self-hosted / GPU path**: needs a GPU node pool (Ray GPU worker group), model
  weights for a text-to-video model (e.g. Stable Video Diffusion, ModelScope T2V,
  AnimateDiff), and a Ray actor loading them once and serving `generate` calls. Wire the
  actor handle's `.generate.remote(...)` as the `model_runner` passed to
  `SelfHostedVideoProvider`.
- **External-API path**: needs a provider account + API key, registered per-tenant via
  `POST /video-providers` (encrypted at rest). `ReplicateVideoProvider.api_base`
  defaults to `https://api.replicate.com/v1` but is configurable per `VideoProviderConfig`.
- Neither is exercised beyond mocks in this repo/environment.

## Out of scope for C5

Directive-driven provider selection beyond "first enabled config" (bias/edge-case
steering of *which provider*, as opposed to prompt content) — reuse the same
`spec.directives` dict already threaded into the prompt; richer selection policy is a
C6/follow-up concern. fal.ai/Runway-specific adapters (the abstraction supports them;
only one external adapter is built here, as scoped). Web UI for browsing/downloading
video manifests — the existing `/datasets/{id}/versions` + `/versions/{v}/download`
routes already work generically over any `DatasetVersion` (including
`format="video-manifest"`); a manifest-aware UI viewer is a nice-to-have, not required
for this spec's "correct abstraction + wiring" bar.
