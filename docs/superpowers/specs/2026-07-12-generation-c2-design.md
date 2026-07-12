# Anodyne — Generation C2 (Text Generation) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C2
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) ·
  [Generation C0](./2026-07-12-generation-c0-design.md) (merged: `anodyne-dataset` domain model,
  `Generator`/`SchemaProposer` ports, Temporal `GenerationWorkflow` + activities, Ray shard execution,
  gateway dataset endpoints — all reused, not rewritten)

## Goal

Generate **text-modality** datasets — classification, Q&A, summarization, and chat/instruction
corpora — via the tenant's registered LLM (`anodyne-llm`), through the same Temporal + Ray pipeline
C0 built for tabular. A user creates a `DatasetSpec` with `modality="text"`, the schema (fields)
describes the row shape, `directives` steer content (topic/tone/label balance), and generation
produces deduplicated, quality-filtered rows written as JSONL + a manifest.

## Decisions (from architecture spec + this design)

| Decision | Choice |
|---|---|
| New code location | New package `anodyne-text` (ports: `anodyne-dataset`/`anodyne-core` only). **`anodyne-generation` is not modified** — a parallel effort (C1, tabular-full) owns it; `anodyne-text` imports from it only where genuinely useful (none currently — text generation doesn't reuse `TabularSampler`/`LLMSchemaProposer` internals). |
| Row-shape selection | No new `DatasetSpec` field. The generator infers the shape from the **names of `spec.fields`**: `{text, label}` → classification, `{question, answer}` → Q&A, `{document, summary}` → summarization, `{instruction, response}` (or a single `messages` field) → chat/instruction. Anything else → a generic structured-row template. This keeps `anodyne-dataset` untouched (no risk of colliding with C1) and mirrors C0's own field-driven design. |
| Chat rows | Modeled as a flat field (`instruction`/`response` pair, the common SFT shape) or a single `messages` text field holding a JSON-encoded turn list — both are plain string/JSON columns, so they fit the existing `pyarrow.Table` shard contract unchanged. |
| Structured output | Each LLM call is asked for a JSON array of row objects with exactly the target field names; parsed with `json.loads` + per-row validation (required keys present, values are strings) via a small Pydantic-free check (mirrors `LLMSchemaProposer`'s fenced-code-block extraction, without depending on it). |
| Determinism | LLM content is **not** byte-deterministic (unlike C0's Faker-based `TabularSampler`) — the architecture spec only requires determinism for the tabular sampler. `seed` still drives: (a) the `seed` request param passed to the LLM (respected by providers that support it), (b) which shard/batch index is requested (so re-running the same shard asks for the same *slice* of rows), and (c) Python's own `random` used for directive-driven decisions (e.g. label-balance assignment order). Documented, not ambiguous — no escalation needed. |
| Batching | One LLM call requests a bounded batch (default 20 rows, configurable via `directives["batch_size"]`) rather than one call per row, to control cost/latency. A shard loops batches until its row count is met or `directives["max_attempts"]` (default 5) batches have been spent, at which point it returns whatever valid rows it has (a partial shard) rather than failing the whole job — logged via the row count actually produced. |
| Deduplication | Exact-match dedup on the shape's primary text field (`text`/`question`/`document`/`instruction`) within a shard's accumulated rows, via a seen-hash set. (Cross-shard dedup is a documented non-goal — see below.) |
| Quality filtering | Drop rows where the primary text field is empty/whitespace-only, or its length falls outside `directives["min_length"]`/`["max_length"]` (defaults 1 / 20,000 chars). Filtering happens before dedup so short-circuited empty strings never count as "seen". |
| Generator port fit | `Generator.generate(spec, start_row, count, seed) -> pyarrow.Table` is synchronous (C0's port, unchanged). `TextGenerator` (constructed with an `LLMProvider` + `ModelConfig`, mirroring `LLMSchemaProposer`) implements it by driving the async `LLMProvider.complete` via `asyncio.run(...)` internally — the same pattern already used for schema proposal, just invoked per-shard instead of per-dataset. |
| Secrets over Ray | The Ray remote task receives the tenant's `ModelConfig` **with its `secret_ref` still encrypted** (never a decrypted API key) plus the raw Fernet `secret_key` string; it builds its own `FernetSecretStore`/`LiteLLMProvider` in-process and decryption happens at LLM-call time — identical to how `LiteLLMProvider._kwargs` already defers decryption today. No plaintext secret ever crosses the Ray wire. |
| Orchestration wiring | `anodyne_workflows.activities.generate_shards` dispatches on `spec.modality`: `TABULAR` keeps calling `anodyne_compute.remote_generate_shard` (byte-for-byte unchanged path); `TEXT` calls a new `anodyne_compute.remote_generate_text_shard`, resolving the model config via a small `model_registry` added to `ActivityContext`. `plan_shards` becomes shape-aware too: text shards are sized much smaller (default 200 rows) than tabular's 50,000, since each row costs an LLM call. Both changes are additive (new optional `ActivityContext`/`GenerationInput` fields with safe defaults) — existing tabular tests and behavior are unaffected. |
| Artifact format | Tabular keeps producing `artifact.parquet` (unchanged). Text produces `artifact.jsonl` (one JSON object per row — the conventional shape for LLM/SFT corpora) **and** a sibling `manifest.json` (task shape, field names, rows requested/produced, dedup + quality-filter counts, model config id, seed) written by the same `assemble_and_upload` activity, now modality-branching at the final-write step only; shard-level bytes stay uniformly Parquet-encoded internally (an implementation detail, not the public artifact) so the shard/assemble plumbing is shared code, not duplicated. `DatasetVersion.format` is set to `"jsonl"` for text. |
| Gateway | Reuses `POST /datasets` (add `modality: Modality = Modality.TABULAR` to the request; schema proposal is already modality-agnostic field/type inference, so no proposer changes) and `POST /datasets/{id}/generate` (add optional `model_config_id` to the request; the route resolves — explicit id, else the tenant's first registered model — and 400s clearly if none is registered, mirroring `get_schema_proposer`'s existing behavior). No new RBAC permissions: `datasets:read`/`datasets:write` already gate every route touched. |
| Web UI | Out of scope for C2 (the architecture roadmap treats UI as incremental "H", and only C0 calls out a UI deliverable explicitly). |

## Components

### 1. `anodyne-text` (new package)

- `errors.py` — `TextGenerationError`.
- `shapes.py` — `TextShape` enum (`CLASSIFICATION`, `QA`, `SUMMARIZATION`, `CHAT`, `GENERIC`);
  `detect_shape(fields: list[FieldSpec]) -> TextShape` (pure, name-set matching) and
  `primary_field(shape, fields) -> str` (which field dedup/quality-filtering keys on).
- `prompts.py` — `build_batch_prompt(spec, shape, batch_size, seed, batch_index) -> list[Message]`:
  a system message describing the exact JSON-array-of-objects contract for the shape + field names/
  types, folding in `spec.description` and `spec.directives` (topic, tone, label balance/choices)
  as steering text; a user message asking for `batch_size` rows.
- `quality.py` — `passes_quality(row: dict, primary: str, min_length: int, max_length: int) -> bool`;
  `Deduplicator` (seen-hash set + `is_duplicate(row, primary) -> bool`).
- `generator.py` — `TextGenerator(Generator)`: constructed with `(provider: LLMProvider, model_config:
  ModelConfig)`. `generate(spec, start_row, count, seed)`:
  1. detect shape + primary field,
  2. loop: build a batch prompt (batch index = start_row // batch_size + attempt), call
     `provider.complete`, parse the JSON array (fenced-block-tolerant like `LLMSchemaProposer`),
     validate each row has all target field names as strings, quality-filter, dedup,
  3. stop when `count` valid rows collected or `max_attempts` batches exhausted,
  4. return a `pyarrow.Table` with exactly `spec.fields` columns (truncated/padded to `count` rows —
     a short shard is valid, never silently duplicated to pad).
  Raises `TextGenerationError` only on unrecoverable failures (e.g. the model never returns valid
  JSON in any attempt); a partially-filled shard is not an error.

### 2. `anodyne-compute` extension

- `ray_tasks_text.py`: `generate_text_shard_bytes(spec, model_config, secret_key, start_row, count,
  seed) -> bytes` (builds `FernetSecretStore` → `LiteLLMProvider` → `TextGenerator` in-process,
  serializes the resulting table as Parquet bytes — shard-level wire format stays Parquet
  regardless of final artifact format, see above) and `@ray.remote remote_generate_text_shard`.
  Exported from `anodyne_compute.__init__`. New deps: `anodyne-text`, `anodyne-llm`, `anodyne-core`,
  `anodyne-storage` (for `FernetSecretStore` only).

### 3. `anodyne-workflows` extension

- `workflow.py`: `GenerationInput` gains `model_config_id: str | None = None` (unused by the
  workflow itself — just carried through to activities, like `seed` already is).
- `activities.py`:
  - `ActivityContext` gains `model_registry: ModelRegistryLike | None = None` and
    `secret_key: str = ""` (both optional/defaulted — existing `ActivityContext(repo=..., s3_bucket=...,
    s3_client=...)` call sites are unaffected).
  - `plan_shards` now fetches `spec = await ctx.repo.get_spec(...)` and sizes shards at 50,000 rows
    for `TABULAR`, 200 rows for `TEXT` (`_TEXT_SHARD_ROWS`).
  - `generate_shards` dispatches per `spec.modality`: `TABULAR` → existing
    `remote_generate_shard` call (byte-identical to today); `TEXT` → resolves the `ModelConfig` via
    `ctx.model_registry.get(tenant_id, model_config_id)` (raises `ValueError` — surfaced as an
    activity failure/retry — if missing) and calls `remote_generate_text_shard`.
  - `assemble_and_upload` reads the spec's modality: `TABULAR` writes `artifact.parquet` (unchanged
    return value/behavior); `TEXT` converts the concatenated table to JSONL, writes
    `artifact.jsonl` + a `manifest.json` (shape, fields, rows produced, model_config_id, seed),
    and returns the `.jsonl` key.
  - `register_version` passes `format="jsonl"` when the spec modality is `TEXT` (else `"parquet"`,
    unchanged default).

### 4. `apps/generation-worker` extension

- `WorkerDeps` gains `model_registry: ModelRegistryLike | None = None` and `secret_key: str = ""`.
- `build_worker` forwards them into `ActivityContext`.
- `main()` constructs `SqlModelRegistry(engine, FernetSecretStore(...))` and passes
  `settings.secret_key` (already an existing `Settings` field, currently unused there).

### 5. `apps/api-gateway` extension

- `CreateDatasetRequest.modality: Modality = Modality.TABULAR`; `create_dataset` uses it instead of
  the hardcoded `Modality.TABULAR`.
- `GenerateRequest.model_config_id: UUID | None = None`.
- `start_generation` gains `registry: ModelRegistry = Depends(deps.get_model_registry)` (already
  used elsewhere in the gateway; consistent with `get_schema_proposer`'s existing requirement that
  a valid `ANODYNE_SECRET_KEY` + at least one registered model exist for dataset workflows). For
  `spec.modality == Modality.TEXT`: resolve `model_config_id` (explicit body value, validated
  against the tenant via `registry.get`; else the tenant's first registered config) and 400 clearly
  if none exists. For `TABULAR`, behavior is unchanged (registry is injected but not consulted).

## Testing strategy (TDD, no shortcuts)

- **Unit (`anodyne-text`):** shape detection for all four shapes + generic fallback; prompt builder
  includes field names + directives; quality filter (empty/too-short/too-long); dedup; `TextGenerator`
  against a **mocked `LLMProvider`** (valid batch, fenced-JSON batch, one malformed batch followed by
  a valid retry, dedup-across-batches reducing yield, `TextGenerationError` when every attempt fails).
- **Unit (`anodyne-compute`):** `generate_text_shard_bytes` against a mocked/fake `LLMProvider`
  wired through a stub secret store — no real network; Ray-remote parity test marked `integration`
  (matches the existing `test_ray_tasks.py` pattern).
- **Unit (`anodyne-workflows`):** `plan_shards` shard-size branches by modality (fake repo returning
  a text vs. tabular spec); `generate_shards` dispatch (patch/fake the two remote-call sites);
  `assemble_and_upload` JSONL + manifest output for text, unchanged Parquet output for tabular;
  `register_version` format branch.
- **Unit (`apps/generation-worker`):** `build_worker` wiring still registers all 5 activities +
  1 workflow (regression); new `WorkerDeps` fields threaded through.
- **Unit (`apps/api-gateway`):** `POST /datasets` with `modality: "text"`; `POST /generate` resolves
  `model_config_id` (explicit / default-first / 400-when-none) for text specs, and is a no-op change
  for tabular specs (existing tests keep passing unmodified except the fixture gaining a
  `get_model_registry` override, mirroring the existing fixture's per-dependency-override pattern).
- All LLM interaction is mocked — no test hits a real model. Docker/Ray/Temporal-server-requiring
  tests are marked `integration`, matching C0.

## Definition of done

A tenant with a registered model creates a `modality="text"` `DatasetSpec` (e.g. fields `{text,
label}` + directives `{"topic": "customer support", "labels": ["urgent","normal"]}`), starts
generation, and the existing Temporal workflow + Ray shard execution produces a deduplicated,
quality-filtered JSONL artifact + manifest, downloadable via the existing presigned-URL route.
Tabular generation is provably unaffected (existing C0 tests all still pass unmodified in behavior).
Non-integration suite green; `ruff`/`mypy --strict` clean; new package registered in root
`pyproject.toml` + `uv.lock` regenerated.

## Non-goals (C2)

Cross-shard/cross-job deduplication; from-template or from-sample sources for text (only
`source="description"`, matching C0's tabular scope); fine-grained per-provider `seed` guarantees
(best-effort passthrough only); a text-specific Web UI; image/audio/video (C3–C5); bias/edge-case
directive catalog beyond free-form `directives` steering (C6).
