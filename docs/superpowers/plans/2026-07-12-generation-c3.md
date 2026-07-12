# Generation C3 — Image Generation Implementation Plan

**Goal:** provider-agnostic image dataset generation: `ImageProvider` port + one external adapter
(OpenAI Images) + one self-hosted adapter interface (SDXL-shaped, Ray-GPU-actor-ready), deterministic
prompt derivation from `DatasetSpec`/directives, wired through Temporal + Ray as `modality="image"`,
manifest + image files in object storage, per-tenant encrypted provider config, RBAC. All tests mock
`ImageProvider` — no GPU, no live API calls.

**Global constraints:** same as C0 (`docs/superpowers/plans/2026-07-12-generation-c0.md`): Python
3.12, uv workspace, register every new package in root `pyproject.toml` + regenerate `uv.lock`,
`ruff`/`mypy --strict` clean, `pytest -m "not integration and not e2e"` green after every task,
globally-unique test basenames prefixed `test_image_*`, no `tests/__init__.py`, Docker/Ray tests
marked `integration`, conventional commits per task.

---

### Task 1: `anodyne-image` — models, ports, errors

**Create:** `packages/anodyne-image/pyproject.toml`, `src/anodyne_image/__init__.py`,
`models.py`, `ports.py`, `errors.py`; `tests/test_image_models.py`.
**Modify:** root `pyproject.toml`.

- `GeneratedImage(data: bytes, mime_type: str)`, `ImagePromptItem(item_index: int, label: str | None, prompt: str)`, `ImageManifestEntry(item_index, object_key, prompt, label, mime_type)`.
- `ImageProvider(ABC)`: `async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage`.
- `ImageProviderError(Exception)`.
- Tests: model defaults/field values; `ImageProvider` is abstract (can't instantiate; a minimal subclass can).
- Deps: `anodyne-core`, `anodyne-dataset`, `pydantic>=2.8`.

### Task 2: `anodyne-image` — `ImagePromptBuilder`

**Create:** `src/anodyne_image/prompts.py`; `tests/test_image_prompts.py`.

- `build(spec, start_row, count) -> list[ImagePromptItem]`: label rotates through the first
  `CATEGORICAL` field's `constraints["choices"]` (fallback `spec.directives["labels"]`, fallback
  no label) by `item_index % len(choices)`; prompt composes description + label +
  `directives["bias"/"use_case"/"edge_case"/"style"]`.
- Tests: determinism (two calls, same args ⇒ identical); labels cycle correctly across a shard
  boundary (`start_row=10` continues the same rotation as `start_row=0,count=20` would have at
  index 10); directive text appears in the prompt; no-label spec ⇒ `label is None` and prompt still
  well-formed; disjoint ranges produce different `item_index`s but the *same* prompt text is
  reproducible for the same `item_index` regardless of which shard call produced it.

### Task 3: `anodyne-image` — `OpenAIImageProvider` (external adapter)

**Create:** `src/anodyne_image/providers/__init__.py`, `providers/openai.py`;
`tests/test_image_provider_openai.py`.

- Constructor: `api_key: str | None`, `api_base: str = "https://api.openai.com/v1"`, `model: str`,
  `params: dict[str, object]`, `http_client` (duck-typed: any object with
  `async def post(url, *, json, headers, timeout) -> response` where `response` has
  `.status_code`/`.json()`/`.text`) — defaults to constructing a real `httpx.AsyncClient` lazily
  (never touched in tests).
- `generate(prompt, *, seed, size)`: POST `{api_base}/images/generations` with
  `{model, prompt, n:1, size, response_format:"b64_json", **params}` + bearer header; decode
  `resp.json()["data"][0]["b64_json"]` via `base64.b64decode` → `GeneratedImage(data=..., mime_type="image/png")`.
- Tests (fake http client, no network): happy path decodes bytes correctly; non-2xx status raises
  `ImageProviderError` with the body in the message; missing `b64_json` key raises
  `ImageProviderError`; request built with the right URL/headers/payload (asserted against the fake
  client's recorded call); no `api_key` configured raises `ImageProviderError` before any network
  call.
- Deps: `httpx>=0.27` added to `anodyne-image`.

### Task 4: `anodyne-image` — `SelfHostedSDXLProvider` + `RayGpuActorPipeline` (self-hosted adapter)

**Create:** `providers/selfhosted.py`; `tests/test_image_provider_selfhosted.py`.

- `DiffusionPipeline` protocol: `def __call__(self, prompt: str, *, seed: int, size: str) -> bytes`.
- `SelfHostedSDXLProvider(pipeline: DiffusionPipeline | None = None)`: `generate()` runs the pipeline
  via `asyncio.to_thread` (keeps the port async without blocking the event loop on what would be a
  synchronous GPU call); raises `ImageProviderError` with a clear "requires a GPU node... inject a
  pipeline" message when `pipeline is None`.
- `RayGpuActorPipeline(actor_handle)`: `__call__` does `ray.get(actor_handle.generate.remote(prompt, seed, size))` — adapts a Ray actor handle into the `DiffusionPipeline` callable.
- Tests: stub pipeline (plain Python callable returning fixed bytes) ⇒ correct `GeneratedImage`;
  no pipeline ⇒ `ImageProviderError`; `RayGpuActorPipeline` against a **local-mode Ray actor**
  (marked `integration`, mirrors `test_ray_tasks.py`) that just echoes back fixed bytes — proves the
  Ray-GPU-actor wiring shape without needing a GPU or `diffusers`.

### Task 5: `anodyne-image` — provider factory + `ImageGenerator`

**Create:** `factory.py`, `generator.py`; `tests/test_image_factory.py`, `tests/test_image_generator.py`.

- `factory.py`: `register_provider(name, ctor: Callable[[ModelConfig, str | None], ImageProvider])`,
  `resolve_image_provider(config, api_key) -> ImageProvider`; pre-registers `"openai-images"` and
  `"sdxl-self-hosted"`; unknown provider ⇒ `ImageProviderError`.
- `generator.py`: `ImageGenerator(Generator)` binds a provider + `ImagePromptBuilder`; `generate(spec, start_row, count, seed)` builds prompts, drives `provider.generate(...)` per item via `asyncio.run`, returns a `pa.Table` with columns `item_index, label, prompt, image_bytes (binary), mime_type`.
- Tests: factory dispatch (fake `ModelConfig.provider` strings) + unknown-provider error;
  `ImageGenerator` — deterministic table given a deterministic fake provider (same seed/range twice
  ⇒ `.equals()`); correct column names/types (`image_bytes` is `pa.binary()`); `count` rows produced;
  disjoint ranges produce disjoint `item_index` values; a raising fake provider propagates the error
  (no silent row-dropping).

### Task 6: `anodyne-storage` — `image_provider_configs` table + migration + `SqlImageProviderRegistry`

**Modify:** `anodyne-storage/src/anodyne_storage/db.py` (new table + `_TENANT_TABLES` entry).
**Create:** migration `migrations/versions/0003_image_provider_configs.py` (mirrors `0002_datasets.py`, `down_revision="0002"`); `src/anodyne_image/registry.py`
(`SqlImageProviderRegistry`, in `anodyne-image` — mirrors `anodyne_llm.registry.SqlModelRegistry`,
imports the new table + `tenant_session` from `anodyne_storage.db`); test
`packages/anodyne-image/tests/test_image_provider_registry.py` (marked `integration`,
testcontainers Postgres, mirrors `anodyne-storage/tests/test_dataset_repo.py`).

- Columns identical to `model_configs`. Kept as a *separate* table from `model_configs` (see spec
  rationale) — nothing in the existing LLM registry/gateway flow changes.
- Registry: `create(tenant_id, *, name, provider, model, api_key, api_base, params) -> ModelConfig`,
  `get`, `list`, `delete` — identical shape/semantics to `SqlModelRegistry`.
- Tests: tenant isolation (create under t1, invisible to t2); secret_ref round-trips via
  `FernetSecretStore` (never plaintext in the row); list/delete.
- Deps: `anodyne-image` gains `anodyne-storage`, `cryptography` (via `anodyne-storage`, already
  transitively present — add explicit dep for `FernetSecretStore` type imports if needed).

### Task 7: `anodyne-compute` — image Ray shard task

**Create:** `src/anodyne_compute/image_tasks.py`; `tests/test_image_ray_tasks.py` (marked
`integration`, mirrors `test_ray_tasks.py` exactly: local-mode Ray, a fake `ImageProvider`
registered into the factory for the duration of the test).

- `generate_image_shard_bytes(spec, start_row, count, seed, provider_config, api_key) -> bytes`:
  `resolve_image_provider` → `ImageGenerator(...).generate(...)` → Parquet bytes.
- `@ray.remote def remote_generate_image_shard(...)`.
- Tests: local call and `ray.get(remote_...remote(...))` produce identical bytes (mirrors
  `test_ray_remote_matches_local`); resulting Parquet round-trips to the same row/column shape as
  `ImageGenerator` produces directly.
- Deps: `anodyne-compute` gains `anodyne-image`.

### Task 8: `anodyne-workflows` — modality-dispatch wiring (minimal, scoped)

**Create:** `src/anodyne_workflows/image_activities.py`;
`tests/test_image_activities.py` (non-integration: fakes for repo/object-store, a fake image
registry + fake `SecretStore`; the Ray call itself is monkeypatched to a fake — no Ray/Docker
needed here, mirroring how `test_activities.py` covers `assemble_and_upload`/`register_version`
without touching `generate_shards`'s real Ray path).
**Modify:** `workflow.py` (`GenerationInput.modality: str = "tabular"` — additive field only, `run()`
untouched); `activities.py` (`ActivityContext` gains `image_registry: ImageConfigRegistry | None = None`,
`secret_store: SecretStore | None = None`; `generate_shards`/`assemble_and_upload`/`register_version`
each gain one `if modality == "image": return await image_activities....` branch at the top, tabular
path below unchanged).

- `image_activities.generate_image_shards(inp, shards, spec, store, provider_config, api_key) -> list[str]`
  — per shard: `remote_generate_image_shard.remote(...)` → `ray.get` (via `asyncio.to_thread`,
  same as the tabular path) → upload shard Parquet → collect keys.
- `image_activities.assemble_image_manifest(inp, keys, store) -> str` — read every shard, write each
  row's `image_bytes` out as its own object (`.../images/{item_index}.png`), build+upload
  `manifest.json`, return its key.
- `activities._resolve_image_provider_config(ctx, tenant_id) -> tuple[ModelConfig, str | None]` —
  `ctx.image_registry.list(tenant_id)` → first config; decrypts `secret_ref` via `ctx.secret_store`
  if both present.
- `register_version`: `format = "image_manifest" if inp.modality == "image" else "parquet"`.
- Tests: `generate_shards`/`assemble_and_upload`/`register_version` all exercised with
  `modality="image"` end-to-end (moto S3 + fakes), producing a `manifest.json` with the right
  entries and a `DatasetVersion(format="image_manifest")`; no tenant configured ⇒ clear
  `ValueError`/`ImageProviderError` (not a silent empty result); the *tabular* path (existing tests
  in `test_activities.py`, `test_workflow.py`) is re-run unchanged to confirm zero regression.

### Task 9: `apps/generation-worker` — wire image registry + secret store

**Modify:** `config.py` (`secret_key: str = ""`); `main.py` (`WorkerDeps` gains
`image_registry`/`secret_store` optional fields, `build_worker` passes them into
`ActivityContext`, `main()` constructs `SqlImageProviderRegistry` + `FernetSecretStore` when a
key is configured).
**Modify:** `tests/test_worker_wiring.py` — add a case asserting `build_worker` wires the new
context fields when supplied (still passes with them omitted — backward compatible).

### Task 10: `apps/api-gateway` — image-provider CRUD + image dataset creation + RBAC

**Modify:** `anodyne_tenancy/authz.py` (`image_providers:read/write/delete`); its test.
**Modify:** `deps.py` (`get_image_provider_registry`), `app.py` (new routes below),
`config.py` if needed.
**Create/modify test:** `apps/api-gateway/tests/test_image_dataset_routes.py`.

- `POST /image-providers` / `GET /image-providers` / `DELETE /image-providers/{id}` — mirrors
  `/models`, RBAC-gated, never returns `secret_ref`.
- `POST /datasets/image` — body `{name, description, target_count, labels: list[str] = [], directives: dict = {}}`;
  builds `DatasetSpec(modality=IMAGE, source="description", fields=[FieldSpec(name="label", semantic_type=CATEGORICAL, constraints={"choices": labels})] if labels else [], target_rows=target_count, directives=directives)`; no LLM schema proposal. RBAC `datasets:write`.
- `start_generation`: guard becomes `if spec.modality is Modality.TABULAR and not spec.fields: raise 400`; `GenerationInput(..., modality=str(spec.modality))`.
- Tests: create an image dataset (with and without labels) → 201 + correct spec; viewer → 403;
  generating an image dataset with zero fields does **not** 400 (regression guard for the tabular
  case still does); `/image-providers` CRUD round-trip + RBAC; existing tabular dataset tests
  (`test_dataset_routes.py`) re-run unchanged.

### Task 11: Full-suite self-review + docs note

- Run full non-integration suite + `ruff` + `mypy --strict` across the repo.
- Where Docker/Ray are available in this environment, run the `integration`-marked tests too;
  otherwise note in the final report which ones require Docker/local-Ray and confirm they're
  correctly marked (not silently skipped from the *non-integration* lane).
- Add a short paragraph to `docs/architecture.md` (or leave a pointer if it's a living index)
  noting C3 landed, provider-agnostic image generation, and the GPU/keys caveat for live runs.
- Commit, push `feat/generation-c3-image`.

---

## Self-review (spec coverage)

`ImageProvider` port + self-hosted/external adapters → T1,T3,T4 ✓. Prompt derivation from
spec/directives → T2 ✓. Per-tenant encrypted provider config (reusing the LLM/SecretStore pattern)
→ T6 ✓. `Generator`-port conformance, worker dispatch by `spec.modality` → T5,T7,T8 ✓. Temporal +
Ray wiring, minimal/scoped shared-file touch → T8 ✓. Manifest + image files in object storage → T8
✓. RBAC + gateway routes → T10 ✓. All tests mock `ImageProvider`/use local-Ray only, no GPU/keys →
every task's test list ✓. New package registered in root `pyproject.toml` → T1 (and re-verified
whenever a new package dep is added in T6/T7).
