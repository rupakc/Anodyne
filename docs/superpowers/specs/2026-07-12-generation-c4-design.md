# Anodyne — Generation C4 (Audio Generation) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C4
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) · [Generation C0](./2026-07-12-generation-c0-design.md) (foundation + tabular slice, on `main`)

## Goal

Provider-agnostic audio dataset generation: synthesize labeled audio clips (TTS from text
prompts, or labeled speech samples) driven by a `DatasetSpec` with `modality = audio`, store the
clips + a JSON manifest in the object store, and wire the `audio` path through the same
Temporal workflow + Ray/worker plumbing C0 established for tabular — **without** rewriting the
tabular path and **without** requiring a GPU or a live provider API key to develop or test.

## Constraints specific to this environment

This worktree has no GPU and no provider credentials. Every unit test **mocks** the
`AudioProvider` port — no real model inference, no live HTTP call, no network access. The
self-hosted (GPU) adapter and Ray actor are built to the correct interface and documented, but
are only exercised (marked `integration`) with fakes standing in for the model/Ray runtime. A
live run needs a GPU node pool (self-hosted OSS TTS) and/or a registered provider API key
(external API), configured exactly like `anodyne-llm` model configs today.

## Decisions

| Decision | Choice |
|---|---|
| Port location | `AudioProvider` (+ `AudioSynthesisRequest`/`AudioSynthesisResult`) added to `anodyne_dataset.{models,ports}` — additive, alongside the existing `Generator`/`SchemaProposer` ports. Ports live with the domain, adapters live in a new package. |
| New package | `anodyne-audio` — `AudioDatasetGenerator` (orchestration: plan text/label items from the spec, call the provider, produce manifest rows) + two adapters: `ElevenLabsAudioProvider` (external API) and `SelfHostedAudioProvider` (self-hosted OSS TTS, e.g. XTTS/Bark, via an injected synth callable). |
| Self-hosted / GPU seam | `SelfHostedAudioProvider` takes an injected `async (text, voice) -> bytes` callable, so the adapter itself has **no** Ray/GPU dependency and is unit-testable with a plain async fake. Production wiring (`apps/generation-worker`) supplies a callable that calls a Ray remote GPU actor — `anodyne_compute.audio_actor.SelfHostedTTSActor` — added additively next to the existing tabular Ray task. The actor lazily imports/loads the real model so importing the module never requires GPU packages. |
| External API adapter | `ElevenLabsAudioProvider` — `POST /v1/text-to-speech/{voice_id}` (per ElevenLabs docs: JSON body `{text, model_id, voice_settings}`, `xi-api-key` header, raw audio bytes response) via `httpx.AsyncClient`. Tests use `httpx.MockTransport` — no real network. |
| Per-tenant provider config | **Reuses** `anodyne-llm`'s existing model-registry pattern verbatim: `ModelConfig` (`provider`, `model`, `params`, encrypted `secret_ref`, `api_base`) rows in the existing `model_configs` table (already tenant-scoped + RLS), via the existing `SqlModelRegistry`. An audio provider is just a `ModelConfig` with `provider="elevenlabs"` (external) or `provider="xtts"`/`"bark"`/`"selfhosted"` (self-hosted). No new table. |
| Steering (`directives`) | `DatasetSpec.directives["audio"]` (free-form JSON, no schema change): optional `prompts: list[str]`, `labels: list[str]`, `voice: str`, `language: str`, `model_config_id: UUID`. Missing `prompts` fall back to deterministic Faker sentences (seeded), mirroring `TabularSampler`'s `TEXT` fallback — so "generate N audio items" works with zero directives. |
| Modality dispatch (worker) | `anodyne_workflows.activities.generate_shards` / `assemble_and_upload` / `register_version` each fetch the `DatasetSpec` (already done in `generate_shards`; added to the other two) and branch on `spec.modality`: `AUDIO` → the new manifest-building path; anything else → the existing, **unmodified** tabular path. This is the "`Generator` selected by `spec.modality`" dispatch point; it is intentionally a single `if` per activity so parallel modality specs (C1–C3, C5) land their own branch with a small, mergeable diff. `ActivityContext` gains one new optional field (`audio_provider_factory`, default `None`) — fully backward compatible with existing tabular-only callers/tests. |
| Artifact shape | Audio items are individual files (`datasets/{dataset}/{job}/audio/item-{i}.{format}`); each shard uploads a small manifest *fragment* (JSON list of item metadata) mirroring how tabular shards upload Parquet fragments; `assemble_and_upload` merges fragments into one `manifest.json` (`DatasetVersion.format = "audio_manifest"`). |
| Gateway | Additive `POST /datasets/audio` (reuses `datasets:write`/`datasets:read` RBAC — no new permission needed) builds an audio `DatasetSpec` directly (no LLM schema proposal — the "schema" is one `transcript` field) and stores `directives["audio"]`. `POST /datasets/{id}/generate`, `GET /jobs/{id}`, `WS /jobs/{id}/stream`, `GET .../versions`, `GET .../download` are already modality-agnostic and work unchanged. |

## Components

### 1. `anodyne_dataset` (additive)
- `models.py`: `AudioSynthesisRequest` (`text`, `voice: str | None`, `language: str | None`),
  `AudioSynthesisResult` (`audio_bytes: bytes`, `format: str = "wav"`, `duration_seconds: float | None`).
- `ports.py`: `AudioProvider.synthesize(request) -> AudioSynthesisResult` (async).

### 2. `anodyne-audio` (new package)
- `models.py`: `AudioManifestItem` (`index`, `object_key`, `text`, `label`, `voice`, `format`,
  `duration_seconds`), `AudioManifest` (`dataset_id`, `job_id`, `items`).
- `generator.py`: `AudioDatasetGenerator(provider)` — `plan_items(spec, start_row, count, seed)`
  builds one `AudioItemPlan` (index + `AudioSynthesisRequest` + label) per row from
  `spec.directives["audio"]`, falling back to seeded Faker sentences; `generate(...)` calls
  `provider.synthesize` concurrently (`asyncio.gather`) and pairs results back to their plans.
- `providers/elevenlabs.py`: `ElevenLabsAudioProvider(api_key, voice_id, model_id=..., http_client=...)`.
- `providers/selfhosted.py`: `SelfHostedAudioProvider(synthesize_fn, format="wav", model_name=...)`.

### 3. `anodyne_compute` (additive)
- `audio_actor.py`: `@ray.remote class SelfHostedTTSActor` — lazy `load_model` callable injected at
  construction (production supplies one that loads XTTS/Bark onto GPU); `synthesize(text, voice) ->
  bytes`. Not exercised without Ray/a model loader; integration-marked test uses a stub loader.

### 4. `anodyne_workflows.activities` (modified additively)
- `ActivityContext` gains `audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None`.
- `generate_shards`: audio branch → `AudioDatasetGenerator(await ctx.audio_provider_factory(spec))`
  per shard range; uploads each item + a manifest-fragment key; returns fragment keys.
- `assemble_and_upload`: audio branch → download + merge manifest fragments → `manifest.json`.
- `register_version`: sets `DatasetVersion.format = "audio_manifest"` when `spec.modality is AUDIO`.
- Tabular behavior is byte-for-byte unchanged when `spec` is `None` or `modality != AUDIO`
  (verified: existing `test_activities.py` fakes return `get_spec() -> None`, so they exercise
  exactly the pre-existing tabular path).

### 5. `apps/generation-worker` (modified additively)
- `audio.py`: `AudioProviderFactory(registry, secrets).build(spec) -> AudioProvider` — resolves
  the tenant's `ModelConfig` (via `directives["audio"]["model_config_id"]` or the tenant's first
  audio-capable config), maps `provider == "elevenlabs"` → `ElevenLabsAudioProvider` (decrypts
  `secret_ref`), else → `SelfHostedAudioProvider` wired to a lazily-created Ray actor handle.
- `main.py` / `WorkerDeps` / `build_worker`: one new optional field (`audio_provider_factory`,
  default `None`) threaded into `ActivityContext`; `main()` builds the real factory when a secret
  key is configured. `registered_activities()` / task queue / workflow registration: **unchanged**.

### 6. `apps/api-gateway` (additive)
- `POST /datasets/audio` — `CreateAudioDatasetRequest {name, description, target_rows, directives:
  {prompts?, labels?, voice?, language?, model_config_id?}}` → `DatasetSpec(modality=AUDIO,
  fields=[FieldSpec(name="transcript", semantic_type=TEXT)], directives={"audio": {...}})`.
  Gated on `datasets:write` (existing permission).

## Testing strategy (TDD)

- **Unit (no infra):** `AudioSynthesisRequest`/`Result` + `AudioProvider` contract;
  `AudioDatasetGenerator` planning (prompts/labels/fallback determinism) and concurrent generation
  against a mock `AudioProvider`; `ElevenLabsAudioProvider` against `httpx.MockTransport` (request
  shape + response bytes, error mapping); `SelfHostedAudioProvider` against an async fake
  `synthesize_fn`; `anodyne_workflows.activities` audio branches against fakes/moto-S3 (mirroring
  the existing `test_activities.py` pattern); `AudioProviderFactory` against a fake registry;
  gateway route against fakes (mirrors `test_dataset_routes.py`).
- **Integration (marked, Docker/Ray):** `SelfHostedTTSActor` Ray plumbing with a stub model
  loader (local Ray, no GPU, no real weights).
- All new test files use globally-unique `test_audio_*` basenames; no `tests/__init__.py`.

## Definition of done

`ruff` + `mypy --strict` clean; full non-integration suite green and growing; an audio
`DatasetSpec` created via `POST /datasets/audio`, generated via the unchanged
`POST /datasets/{id}/generate`, produces (with a mocked `AudioProvider` injected in tests) a
`manifest.json` `DatasetVersion` with `format = "audio_manifest"` and per-item audio object keys.
Tabular vertical slice (C0) behavior is unchanged.

## Non-goals

Real GPU model serving or a live ElevenLabs run (needs a GPU node / API key — documented, not
exercised here); a Web UI audio wizard (out of scope, C0's UI covers tabular only); video (C5);
template catalog / directive UI (C6); perturbation/evaluation of audio (D/F).
