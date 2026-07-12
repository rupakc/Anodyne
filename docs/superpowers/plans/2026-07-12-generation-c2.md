# Generation C2 — Text Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship LLM-driven text-dataset generation (classification/QA/summarization/chat) behind the
C0 `Generator` port, wired through the existing Temporal + Ray pipeline and gateway, producing a
deduplicated, quality-filtered JSONL artifact + manifest.

**Architecture:** New package `anodyne-text` (shape detection, prompt templates, quality/dedup,
`TextGenerator`). Extends (does not rewrite) `anodyne-compute`, `anodyne-workflows`,
`apps/generation-worker`, `apps/api-gateway` with modality-branching, additive changes. Does **not**
touch `anodyne-generation` (parallel C1 effort owns it) or `anodyne-dataset` models (shape is
inferred from field names, not a new spec field).

**Tech stack:** same as C0 (Python 3.12, pydantic v2, pyarrow, temporalio, ray, anodyne-llm/LiteLLM).

## Global constraints

- Register `anodyne-text` in root `pyproject.toml` (`[dependency-groups] dev` + `[tool.uv.sources]`),
  regenerate `uv.lock`, before any test can import it.
- `ruff` + `mypy --strict` clean; `pytest -m "not integration and not e2e"` green after every task.
- Test basenames globally unique — prefix every new test file `test_text_*`.
- No `tests/__init__.py`. No test hits a real LLM — mock `LLMProvider`.
- Docker/Ray/Temporal-server tests marked `integration`.
- Conventional commits, one per task.

---

### Task 1: `anodyne-text` — shape detection + prompt building

**Files:** `packages/anodyne-text/pyproject.toml`, `src/anodyne_text/__init__.py`, `errors.py`,
`shapes.py`, `prompts.py`; tests `tests/test_text_shapes.py`, `tests/test_text_prompts.py`.
Modify: root `pyproject.toml`.

**Interfaces:** `TextGenerationError`; `TextShape` enum; `detect_shape(fields) -> TextShape`;
`primary_field(shape, fields) -> str`; `build_batch_prompt(spec, shape, batch_size, seed,
batch_index) -> list[Message]`.

- [ ] Write failing tests: `detect_shape` returns `CLASSIFICATION` for fields named `{text,label}`,
  `QA` for `{question,answer}`, `SUMMARIZATION` for `{document,summary}`, `CHAT` for
  `{instruction,response}` or `{messages}`, `GENERIC` otherwise (order/extra fields tolerated —
  match on subset, not exact set). `primary_field` returns `"text"`/`"question"`/`"document"`/
  `"instruction"`/first field name for `GENERIC`. `build_batch_prompt` returns a system + user
  `Message` list; system message contains every target field name and, when present, `spec.
  directives["topic"]`/`["tone"]`; user message mentions the batch size.
- [ ] Run → FAIL (`ModuleNotFoundError`).
- [ ] Create package (deps: `anodyne-core`, `anodyne-dataset`, `pyarrow>=17` + workspace sources)
  and implement `shapes.py`/`prompts.py`/`errors.py`.
- [ ] Register in root `pyproject.toml`, `uv sync`, run tests → PASS; `mypy`/`ruff` clean.
- [ ] Commit: `feat(text): add text-shape detection and prompt templates`.

---

### Task 2: `anodyne-text` — quality filtering + deduplication

**Files:** `src/anodyne_text/quality.py`; test `tests/test_text_quality.py`.

**Interfaces:** `passes_quality(row, primary, min_length=1, max_length=20_000) -> bool`;
`Deduplicator` with `is_duplicate(row, primary) -> bool` (stateful, exact-match on normalized —
stripped — primary field value).

- [ ] Write failing tests: empty/whitespace-only primary field fails; too-short/too-long fails;
  within-bounds passes; `Deduplicator` flags an exact repeat (after strip) as duplicate, treats
  distinct values as new, and is per-instance (a fresh `Deduplicator` sees no history).
- [ ] Run → FAIL.
- [ ] Implement `quality.py`.
- [ ] Run → PASS; `mypy`/`ruff` clean.
- [ ] Commit: `feat(text): add quality filtering and deduplication`.

---

### Task 3: `anodyne-text` — `TextGenerator` (the `Generator` port)

**Files:** `src/anodyne_text/generator.py`; test `tests/test_text_generator.py` (mocked
`LLMProvider`). Modify: `anodyne-text/pyproject.toml` (add dep on `anodyne-llm`... actually only
needs `anodyne_core.ports.LLMProvider`/`models`, already a dep via `anodyne-core`).

**Interfaces:** `TextGenerator(Generator)` — `__init__(provider: LLMProvider, model_config:
ModelConfig)`; `generate(spec, start_row, count, seed) -> pyarrow.Table`.

- [ ] Write failing tests with a fake `LLMProvider` (records requests, returns scripted
  `LLMResponse`s):
  - a classification spec + a provider returning one valid JSON-array batch of `count` rows →
    table has exactly `count` rows with `text`/`label` columns.
  - a provider returning a fenced ```json block → parsed the same as raw JSON.
  - a provider whose first batch has 3 duplicate/empty rows mixed with valid ones, second batch
    fills the rest → final table has `count` valid, deduplicated rows (dedup/quality filters
    actually invoked, not just plumbing).
  - a provider that always returns unparseable content → `TextGenerationError` after
    `directives["max_attempts"]` (default 5) tries; the fake records exactly that many calls (no
    silent infinite loop).
  - a provider returning fewer valid rows than `count` even after `max_attempts` batches → returns
    a **short** table (no error, no padding/duplication) — asserts `num_rows < count`.
  - two calls with the same `seed`/`start_row`/`count` against a provider that echoes the prompt's
    batch-index back as a row field show the same requested batch index (proves seed→batch-index
    determinism at the *request* level, not content).
- [ ] Run → FAIL.
- [ ] Implement `generator.py`: shape/primary-field via Task 1, prompt via `build_batch_prompt`,
  `asyncio.run` driving `provider.complete`, JSON parse (fenced-block tolerant, reuse the same
  regex approach as `LLMSchemaProposer` — duplicated locally, not imported, since
  `anodyne-generation` is off-limits), per-row key/type validation, `quality.passes_quality` +
  `Deduplicator`, loop until `count` or `max_attempts`, build `pa.table` from the field list
  (missing optional fields default to `""`).
- [ ] Run → PASS; `mypy`/`ruff` clean.
- [ ] Commit: `feat(text): add LLM-backed TextGenerator implementing the Generator port`.

---

### Task 4: `anodyne-compute` — Ray text-shard task

**Files:** `packages/anodyne-compute/src/anodyne_compute/ray_tasks_text.py`; test
`tests/test_text_ray_tasks.py` (unit part mocks `LLMProvider`; Ray-parity part marked
`integration`). Modify: `anodyne-compute/pyproject.toml` (+`anodyne-text`, `anodyne-llm`,
`anodyne-core`, `anodyne-storage`), `__init__.py` exports, root `pyproject.toml` unaffected (compute
already registered).

**Interfaces:** `generate_text_shard_bytes(spec, model_config, secret_key, start_row, count, seed) ->
bytes`; `@ray.remote remote_generate_text_shard(...)`.

- [ ] Write failing tests: `generate_text_shard_bytes` with a monkeypatched `LiteLLMProvider.complete`
  (patch at the `anodyne_llm.adapter` level so the function under test exercises its own
  `FernetSecretStore`/`LiteLLMProvider` construction) returns valid Parquet bytes with `count` rows;
  Ray-remote parity test (marked `integration`, mirrors `test_ray_tasks.py`) shows
  `ray.get(remote_generate_text_shard.remote(...))` matches the local call given the same
  monkeypatch active in the Ray worker (use `local_mode=True` so the patch applies in-process).
- [ ] Run → FAIL.
- [ ] Implement `ray_tasks_text.py`.
- [ ] Run non-integration subset → PASS; `mypy`/`ruff` clean. Run integration subset if Docker/Ray
  available.
- [ ] Commit: `feat(compute): add Ray text-shard generation task`.

---

### Task 5: `anodyne-workflows` — modality-aware activities

**Files:** Modify `src/anodyne_workflows/workflow.py` (`GenerationInput.model_config_id`),
`src/anodyne_workflows/activities.py` (`ActivityContext` fields, `plan_shards`, `generate_shards`,
`assemble_and_upload`, `register_version`). Test: `tests/test_text_activities.py`. Modify
`anodyne-workflows/pyproject.toml` (+`anodyne-llm` for the `ModelRegistryLike` protocol's
`ModelConfig` type — already available via `anodyne-core`; add `anodyne-text`/`anodyne-compute`
already present).

**Interfaces:** `ActivityContext(repo=..., s3_bucket=..., s3_client=..., publisher=None,
model_registry=None, secret_key="")` (new fields optional, end of the dataclass — existing
positional/keyword call sites unaffected); `plan_shards` shard-size branch; `generate_shards`
modality dispatch; `assemble_and_upload` JSONL+manifest for text; `register_version` format branch.

- [ ] Write failing tests (fakes, no live Temporal/Ray/DB — mirrors `test_activities.py`'s style):
  - `plan_shards` given a fake repo returning a `TEXT` spec with `target_rows=450` produces shards
    of ≤200 rows each (vs. the existing tabular test's ≤50,000); given a `TABULAR` spec, shard
    sizing is unchanged (regression-pins the existing behavior).
  - `generate_shards` for a `TEXT` spec calls the text remote path (monkeypatch
    `anodyne_compute.remote_generate_text_shard` to a stub returning fixed Parquet bytes) and
    raises `ValueError` clearly if `ctx.model_registry` or `inp.model_config_id` is missing;
    for a `TABULAR` spec, behavior/keys/uploaded bytes are byte-identical to the existing test
    (regression).
  - `assemble_and_upload` for a `TEXT` spec's shard keys (moto-mocked S3, real `S3ObjectStore`)
    writes `artifact.jsonl` (one JSON object per row, matching the shard's rows) **and**
    `manifest.json` (contains `"modality": "text"`, the field names, and `rows_produced`), and
    returns the `.jsonl` key; for `TABULAR`, output is unchanged (`artifact.parquet`, same bytes as
    before — regression, reusing the existing test's fixture shape).
  - `register_version` sets `format="jsonl"` for a `TEXT` spec, `"parquet"` for `TABULAR`
    (regression).
- [ ] Run → FAIL.
- [ ] Implement the branches described in the design doc (`ActivityContext.model_registry` typed as
  a small local `Protocol` — `async def get(self, tenant_id, config_id) -> ModelConfig | None`).
- [ ] Run full `anodyne-workflows` + `anodyne-generation`/`anodyne-compute` non-integration suite →
  PASS (existing tabular tests unmodified in assertions, still green); `mypy`/`ruff` clean.
- [ ] Commit: `feat(workflows): dispatch generation activities on dataset modality`.

---

### Task 6: `apps/generation-worker` — wire the model registry + secret store

**Files:** Modify `src/generation_worker/main.py` (`WorkerDeps`, `build_worker`, `main`). Test:
modify/extend `tests/test_worker_wiring.py` (new test in the same file, keep existing ones
untouched) — or add `tests/test_text_worker_wiring.py` if a wholly new basename is cleaner; use the
latter to respect the "prefix with package" convention while not touching the passing C0 file more
than necessary. Modify `generation-worker/pyproject.toml` (+`anodyne-llm`).

**Interfaces:** `WorkerDeps(repo=..., s3_bucket=..., s3_client=..., publisher=None,
model_registry=None, secret_key="")`; `build_worker` forwards the two new fields into
`ActivityContext`.

- [ ] Write failing test: `build_worker` with `WorkerDeps(..., model_registry=<fake>,
  secret_key="k")` produces an `ActivityContext` (inspect via `configure_activities`'s module-level
  state, same technique `test_activities.py` uses, or expose the bound context for assertion) whose
  `model_registry` and `secret_key` match; omitting them defaults to `None`/`""` (regression: the
  existing wiring test's `WorkerDeps(repo=..., s3_bucket=..., s3_client=None)` call keeps working
  unchanged).
- [ ] Run → FAIL.
- [ ] Implement: extend `WorkerDeps`, thread through `build_worker`; `main()` builds
  `SqlModelRegistry(engine, FernetSecretStore(settings.secret_key.encode()))` guarded the same way
  `api_gateway.deps._secret_store` guards an invalid key (reuse the pattern, don't import
  `api_gateway` — a small local helper in `generation_worker/main.py`).
- [ ] Run → PASS; `mypy`/`ruff` clean.
- [ ] Commit: `feat(generation-worker): wire model registry and secret store for text generation`.

---

### Task 7: `apps/api-gateway` — `modality` on create, `model_config_id` on generate

**Files:** Modify `src/api_gateway/app.py` (`CreateDatasetRequest`, `create_dataset`,
`GenerateRequest`, `start_generation`). Test: extend `tests/test_dataset_routes.py`'s `wired()`
fixture with a `get_model_registry` override + add new test cases (keep all existing test bodies
unmodified — regression).

**Interfaces:** `CreateDatasetRequest.modality: Modality = Modality.TABULAR`;
`GenerateRequest.model_config_id: UUID | None = None`; `start_generation` resolves the model config
for `TEXT` specs via `Depends(deps.get_model_registry)`.

- [ ] Write failing tests: `POST /datasets` with `"modality": "text"` persists a `TEXT` spec (fields
  come from the same fake proposer — schema proposal is modality-agnostic); `POST
  /datasets/{id}/generate` on a `TEXT` spec with no `model_config_id` and no registered models →
  400 with a clear message, no workflow started; with a registered model and no explicit
  `model_config_id` → uses the first one (assert `GenerationInput.model_config_id` in the fake
  Temporal client's recorded call); with an explicit `model_config_id` for a model not owned by the
  tenant → 400; all **existing** tabular tests in the file keep passing with their assertions
  unmodified (only the fixture gains the new override).
- [ ] Run → FAIL.
- [ ] Implement the route changes + fixture's `_FakeModelRegistry`.
- [ ] Run full `apps/api-gateway` non-integration suite → PASS; `mypy`/`ruff` clean.
- [ ] Commit: `feat(gateway): add text modality on dataset create and model selection on generate`.

---

### Task 8: Full-suite pass, self-review, docs cross-check

- [ ] `uv sync --all-extras && uv run pytest -q -m "not integration and not e2e"` → green, count
  reported.
- [ ] `uv run pytest -q -m integration` where Docker/Ray available → green (or documented as
  environment-limited, matching C0's own caveat).
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy .` → clean.
- [ ] Self-review against this plan + the design doc: every row of the design's decision table has
  a corresponding implemented behavior + test; no placeholder bodies.
- [ ] Commit any cleanup; push branch `feat/generation-c2-text` to origin (no merge to `main`).

---

## Self-Review

**Spec coverage:** shape detection + prompts → T1 ✓; quality/dedup → T2 ✓; `TextGenerator`
(structured output, batching, dedup, quality, partial-shard-not-error) → T3 ✓; Ray shard execution
with ciphertext-only secrets → T4 ✓; Temporal activity dispatch (shard sizing, generation, JSONL +
manifest assembly, version format) → T5 ✓; worker wiring → T6 ✓; gateway modality + model selection
→ T7 ✓; full verification → T8 ✓. `directives` steering (topic/tone/label balance) flows through
`build_batch_prompt` (T1) into every generated batch (T3).

**Placeholders:** none — every task names the exact files, function signatures, and test
assertions; T5's `ModelRegistryLike` protocol shape is given explicitly.

**Regression discipline:** every task modifying a shared C0 file (activities.py, workflow.py,
main.py, app.py) explicitly calls out that existing tabular test assertions must stay green
unmodified — this is the concrete guard against silently breaking C0 or colliding with the parallel
C1 tabular effort.

**Isolation from C1:** `anodyne-generation` and `anodyne-dataset` are never modified; the only files
touched outside the new `anodyne-text` package are the C0 orchestration/gateway files this spec
explicitly says must change to wire a new modality through — an unavoidable, minimal-surface-area
set of edits (additive fields + one new `if modality is TEXT` branch per function).
