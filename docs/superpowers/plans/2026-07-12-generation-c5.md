# Generation C5 — Video Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement task-by-task.

**Goal:** Provider-agnostic video generation (`modality="video"`): a new `anodyne-video`
package (`VideoProvider` port + self-hosted-adapter interface + one external-API
adapter + per-tenant provider registry), Temporal/worker wiring additive to C0's
tabular pipeline, gateway CRUD for provider configs. Every test mocks the provider —
no GPU, no live network, no media binaries.

**Depends on:** C0 (merged to `main`): `anodyne-dataset`, `anodyne-workflows`,
`anodyne-compute`, `anodyne-storage`, `anodyne-llm` (pattern reference),
`api-gateway`/`generation-worker`.

## Global constraints (same bar as C0)

- Python 3.12+, uv workspace, `src/` layout. New package registered in root
  `pyproject.toml` (`dev` group + `tool.uv.sources`); `uv lock` regenerated.
- `ruff check` + `mypy --strict` clean; `uv run pytest -m "not integration and not e2e"`
  green after every task.
- Test basenames globally unique (`test_video_*.py`); no `tests/__init__.py`.
- Tenant-scoped tables carry `tenant_id` + RLS. Secrets encrypted via `SecretStore`,
  never logged/returned in plaintext.
- Do not modify `anodyne_workflows/activities.py` or its `ActivityContext` — video gets
  its own context/activities in a new file (see design doc "why video doesn't reuse...").
- No real model inference, no live provider API calls, no video bytes committed to git.
  Test fixtures use tiny fixed placeholder bytes (e.g. `b"fake-mp4-bytes"`).

---

### Task 1: `anodyne-video` package — domain models

**Files:** create `packages/anodyne-video/pyproject.toml`,
`src/anodyne_video/__init__.py`, `models.py`; test `tests/test_video_models.py`; modify
root `pyproject.toml`.

- [ ] Write failing tests: `VideoProviderConfig` defaults (`enabled=True`,
  `secret_ref=None`); `VideoGenerationRequest` field defaults; `VideoAsset` round-trips
  `content: bytes`; `VideoManifestItem`/`VideoManifest` construct and `model_dump`.
- [ ] Run — fails (`ModuleNotFoundError`).
- [ ] Implement `models.py` per design doc (pydantic `BaseModel`s, mirrors
  `anodyne_core.models.ModelConfig` / `anodyne_dataset.models`).
- [ ] Register `anodyne-video` in root `pyproject.toml` dev group + `tool.uv.sources`;
  `uv sync`.
- [ ] Green. `ruff` + `mypy --strict` on the new package.

### Task 2: `ports.py` — `VideoProvider` + `VideoProviderRegistry`

**Files:** `src/anodyne_video/ports.py`; test `tests/test_video_ports.py`.

- [ ] Failing test: a minimal fake subclass of `VideoProvider`/`VideoProviderRegistry`
  can be instantiated (ABCs enforce the abstract methods exist); calling an
  unimplemented method on a bare subclass missing a method raises `TypeError` at
  instantiation.
- [ ] Implement both ABCs (async `generate`; async `create/get/list/delete`).
- [ ] Green, lint, types.

### Task 3: self-hosted adapter interface

**Files:** `src/anodyne_video/adapters/__init__.py`, `adapters/self_hosted.py`; test
`tests/test_video_self_hosted_provider.py`.

- [ ] Failing tests: `SelfHostedVideoProvider(runner=fake_sync_fn)` — `await
  provider.generate(config, request)` calls the injected runner off-thread and wraps
  its return bytes into a `VideoAsset` carrying the request's duration/width/height/fps
  /seed and the config's provider/model; runner exceptions propagate as-is (no
  swallowing).
- [ ] Implement: `asyncio.to_thread(self._runner, request)`; docstring on the Ray/GPU
  production wiring shape (per design doc), explicitly noting it is NOT built here.
- [ ] Green, lint, types.

### Task 4: external-API adapter (`ReplicateVideoProvider`)

**Files:** `adapters/external_api.py`; test `tests/test_video_external_api_provider.py`.
Use `context7` if the Replicate REST shape needs checking (create prediction / poll /
fetch output) — otherwise the generic three-step shape documented below is sufficient
since no live calls are made.

- [ ] Failing tests (via `httpx.MockTransport`, no network):
  - POSTs to `{api_base}/predictions` with the model/version + input (prompt + params),
    sending `Authorization: Bearer <decrypted secret>`.
  - Polls `GET {api_base}/predictions/{id}` until `status == "succeeded"` (a fake
    transport can return `"processing"` once then `"succeeded"`), then downloads the
    output URL's bytes into `VideoAsset.content`.
  - Raises a clear `VideoProviderError` (new exception in `ports.py` or
    `external_api.py`) if `status == "failed"`.
- [ ] Implement `ReplicateVideoProvider(VideoProvider)`: `__init__(self, secret_store,
  client: httpx.AsyncClient)`; `generate()` implements the flow; small bounded poll loop
  (max attempts) rather than unbounded, to fail fast in tests and in production.
- [ ] Green, lint, types.

### Task 5: `generator.py` — prompt building + `VideoDatasetGenerator`

**Files:** `src/anodyne_video/generator.py`; test `tests/test_video_generator.py`.

- [ ] Failing tests:
  - `build_video_prompt(spec, index)` is deterministic (same spec+index → same string),
    varies with `index`, and folds in `directives` values (e.g. `{"style": "noir"}`
    appears in the prompt).
  - `VideoDatasetGenerator.generate_items(spec, provider=fake, config, start_index=0,
    count=3, seed=1)` returns 3 `(VideoManifestItem, bytes)` pairs with contiguous
    `index` values `0,1,2`, each item's `prompt` matching `build_video_prompt`, and each
    calls the fake provider exactly once per item.
- [ ] Implement per design doc.
- [ ] Green, lint, types.

### Task 6: `anodyne-storage` — `video_provider_configs` table + migration + SQL registry

**Files:** modify `anodyne_storage/db.py` (+table, +`_TENANT_TABLES` entry); create
`migrations/versions/0003_video_provider_configs.py`; create
`anodyne_video/registry.py` (`SqlVideoProviderRegistry`); test
`packages/anodyne-video/tests/test_video_registry.py` (integration, testcontainers,
mirrors `anodyne_llm/tests/test_registry.py`).

- [ ] Failing integration test: create encrypts the key + RLS isolates tenants (same
  shape as `test_registry.py`'s `test_create_encrypts_key_and_isolates_tenants`), plus
  `enabled` defaults `True` and round-trips through `list`.
- [ ] Add table to `db.py`, add migration (mirrors `0002_datasets.py`).
- [ ] Implement `SqlVideoProviderRegistry` (mirrors `SqlModelRegistry` exactly, add
  `enabled` bool handling like `ModelConfig`).
- [ ] `anodyne-video`'s `pyproject.toml` gains `anodyne-storage`, `anodyne-core`,
  `sqlalchemy` (transitively via anodyne-storage), `httpx` deps.
- [ ] Green (integration lane — run with `-m integration` locally if Docker
  available; otherwise confirm via code review + the non-integration suite staying
  green), lint, types.

### Task 7: `anodyne_workflows` — `GenerationInput.modality` + workflow branch

**Files:** modify `workflow.py`; test additions in
`packages/anodyne-workflows/tests/test_workflow.py` (new test function, existing ones
untouched).

- [ ] Failing test: a new `test_video_workflow_runs_video_activity_sequence` —
  constructs `GenerationInput(..., modality="video")`, registers fake activities named
  `plan_video_items`/`generate_video_items`/`assemble_video_manifest`/
  `register_video_version`/`set_status`, runs the workflow via `WorkflowEnvironment`
  (integration, mirrors the existing tabular workflow test), asserts the video call
  sequence and that the tabular activities are never invoked.
- [ ] Implement: add `modality: str = "tabular"` to `GenerationInput`; branch in `run()`
  per design doc.
- [ ] Existing `test_workflow_runs_after_approval` (tabular, default modality) still
  green, unmodified.
- [ ] Lint, types.

### Task 8: `anodyne_workflows/video_activities.py` — the four video activities

**Files:** create `video_activities.py`; test `tests/test_video_activities.py` (mirrors
`test_activities.py`: fakes + moto S3, no live infra).

- [ ] Failing tests:
  - `plan_video_items` splits `target_rows` into contiguous `_VIDEO_SHARD_ITEMS`-sized
    shards (same contiguous-coverage property test as `plan_shards`).
  - `generate_video_items`: with a fake `VideoProviderRegistry` (one enabled config)
    and a fake `VideoProvider` in `ctx.providers`, generates the right item count,
    uploads each clip under `datasets/{id}/{job}/videos/item-{i}.mp4` (verify via the
    moto bucket), and returns manifest-item dicts (no raw bytes) with correct
    `object_key`s.
  - `assemble_video_manifest`: given item dicts, uploads
    `datasets/{id}/{job}/manifest.json` containing all items, returns its key.
  - `register_video_version`: writes a `DatasetVersion` with
    `format="video-manifest"`, `artifact_uri=<manifest key>`, `row_count=len(items)`
    (via a fake `DatasetRepository`, same fake class shape as
    `test_activities.py`'s `_FakeDatasetRepository`).
  - No configuration → clear `RuntimeError` (mirrors `_context()` in `activities.py`).
- [ ] Implement per design doc; reuse `_object_store`-style helper (tenant-scoped
  `S3ObjectStore`) analogous to `activities.py`'s but local to this file (no import from
  `activities.py` internals — keep the two activity modules decoupled).
- [ ] Green, lint, types.

### Task 9: `generation-worker` wiring

**Files:** modify `apps/generation-worker/src/generation_worker/main.py`; modify
`apps/generation-worker/tests/test_worker_wiring.py` (extend
`EXPECTED_ACTIVITY_NAMES`, add a fake-video-deps variant of
`test_build_worker_registers...`).

- [ ] Failing test: `registered_activities()` includes the four video activity names in
  addition to the five tabular ones; `build_worker` with `WorkerDeps` that also carries
  fake `secret_store`/`video_registry`/`video_providers` configures both activity
  modules without error.
- [ ] Implement: import the four video activities + `configure_video_activities`,
  `VideoActivityContext`; extend `WorkerDeps` with optional
  `secret_store: SecretStore | None = None`, `video_registry: VideoProviderRegistry |
  None = None`, `video_providers: dict[str, VideoProvider] | None = None`; call
  `configure_video_activities` in `build_worker` (empty dict/`None` deps still wire
  cleanly — the video activities themselves raise clearly if invoked unconfigured);
  wire real `SqlVideoProviderRegistry` + `ReplicateVideoProvider` in `main()`.
- [ ] `apps/generation-worker/pyproject.toml` gains `anodyne-video` dependency.
- [ ] Existing test `test_registered_workflows_and_activities_match_task_queue_constant`
  updated to the grown `EXPECTED_ACTIVITY_NAMES` set (it asserts an exact set — the one
  pre-existing test file this spec must touch, per the design doc).
- [ ] Green, lint, types.

### Task 10: gateway — provider CRUD + modality passthrough

**Files:** modify `anodyne_tenancy/authz.py` (new permissions); modify
`api_gateway/deps.py` (+`get_video_provider_registry`), `api_gateway/app.py`
(+3 routes, +1-line `modality=` fix in `start_generation`); test
`apps/api-gateway/tests/test_video_provider_routes.py` (mirrors
`test_models_routes.py`); one new assertion in `test_dataset_routes.py`'s existing
generate test confirming `inp.modality == "tabular"` for a tabular dataset (back-compat
regression guard).

- [ ] Failing tests: `POST /video-providers` requires `video_providers:write`, encrypts
  the key, never returns `secret_ref`; `GET /video-providers` requires
  `video_providers:read`, strips secrets; `DELETE` requires `:delete` and is
  tenant-scoped; `POST /datasets/{id}/generate` on a `Modality.VIDEO` dataset starts
  the workflow with `GenerationInput.modality == "video"`.
- [ ] Implement per design doc. `_MEMBER` gains `video_providers:read/write`; `_ADMIN`
  gains `video_providers:delete` (mirrors `models:*` placement exactly).
- [ ] `apps/api-gateway/pyproject.toml` gains `anodyne-video` dependency.
- [ ] Green, lint, types.

### Task 11: root wiring, docs, full sweep

**Files:** root `pyproject.toml` (already touched incrementally; verify complete),
`uv.lock` (regenerate), `docs/architecture.md` roadmap row (optional note), this spec +
plan (status).

- [ ] `uv lock` (or `uv sync`) picks up the finished `anodyne-video` package cleanly.
- [ ] Full sweep: `uv run pytest -q -m "not integration and not e2e"` green;
  `uv run ruff check .`; `uv run mypy .` (repo-wide `--strict` per `pyproject.toml`/
  `mypy.ini` config) clean.
- [ ] Self-review pass (see design doc "Out of scope") — confirm no media binaries, no
  live network in tests, `anodyne_workflows/activities.py` diff is empty.
- [ ] Commit, push `feat/generation-c5-video`. Do not merge.
