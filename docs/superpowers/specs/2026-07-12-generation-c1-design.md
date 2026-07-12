# Anodyne — Generation C1 (Tabular, Full / From-Sample) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C1
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) · [C0 design](./2026-07-12-generation-c0-design.md) (merged)

## Goal

Generate high-fidelity tabular data **from an uploaded sample** (CSV/Parquet), and richer
from-description tabular generation, on top of the C0 foundation. A user uploads a sample, the
platform profiles it (schema + per-column distributions + correlations), and generates a
statistically faithful synthetic dataset at scale — Ray-sharded, durably orchestrated by the
existing `GenerationWorkflow`, with from-description generation (`TabularSampler`) untouched.

## Decisions

| Decision | Choice |
|---|---|
| New code location | New package `anodyne-tabular` (hexagonal adapters); `anodyne-generation`'s existing files (`sampler.py`, `proposer.py`) are **not modified** — only imported from, per the parallel-workstream rule. `anodyne-dataset` (the shared ports package) gets **additive-only** changes: two new models, two new ports. No existing port signature changes. |
| Default synthesizer | **Gaussian copula** (`copulas` [MIT] + `rdt` [MIT] for reversible column encode/decode) — fast, deterministic-per-seed, always available, no extra opt-in. |
| Higher-fidelity option | **CTGAN / TVAE** (`ctgan` [MIT], which bundles TVAE) — selectable via `spec.directives["synthesizer"] = "ctgan" \| "tvae"`. Same permissive license as copulas/rdt; heavier (torch-based training), so not the automatic default, but requires no separate opt-in flag. |
| SDV opt-in adapter | `sdv` (BSL 1.1) wrapped in `anodyne_tabular.sdv_adapter`, **deferred-imported** (module loads fine without `sdv` installed) and gated behind an explicit `enable_sdv` flag (worker setting `ANODYNE_TABULAR_ENABLE_SDV`, default `false`) *and* `directives["synthesizer"] = "sdv"`. `sdv` is **not** added to the root `dependency-groups.dev` / `uv.lock` — it stays a true opt-in extra (`anodyne-tabular[sdv]`) so the default install never pulls BSL-licensed code. |
| PII-realistic values | Columns profiled as `name`/`email`/`address`/`text` are **excluded from statistical modeling** and regenerated with Faker (mirroring `TabularSampler`'s per-shard seeding), never leaking real sample values. Faker's `locale` is honored from a field's `constraints["faker_locale"]`; `constraints["provider"] = "mimesis"` switches that one field to Mimesis (satisfies the Faker **and** Mimesis requirement without duplicating every generator). |
| Constraint enforcement | A shared `anodyne_tabular.constraints.enforce()` clips numeric columns to profiled/declared min/max, restricts categoricals to the declared choice set (out-of-vocab values are re-mapped to the most frequent in-vocab value), and nulls out non-nullable-violating cells — applied as a post-processing pass after every synthesizer. |
| Determinism | Statistical/deep synthesizers are stochastic samplers, not per-row-deterministic like `TabularSampler`. We keep the *same seed+range ⇒ same output* contract by reseeding NumPy/Torch's global RNG from `(seed, start_row)` immediately before each shard's `.sample(count)` call — same technique `TabularSampler` uses, applied at the shard-sampling boundary instead of the per-value boundary. Model **fitting** is seeded once per generation job (same seed ⇒ same fitted model, given the same input sample). |
| Fit-once, sample-per-shard | Training/fitting a copula or CTGAN model per Ray shard would be wasteful and would break determinism. The model is fit **once**, synchronously, inside the `generate_shards` activity (off the event loop via `asyncio.to_thread`); the *fitted generator instance* is then shipped to Ray tasks (`anodyne_compute.sample_tasks.remote_generate_shard_from_generator`) — one Ray task per shard, each just calling `.generate()` (`.sample()` + reseed + decode) on the shared fitted model. This is the "Ray-shards large generation" requirement applied to the from-sample path. |
| Schema from sample | `anodyne_tabular.schema.fields_from_profile(profile) -> list[FieldSpec]` mirrors `LLMSchemaProposer`'s role for the from-description path: the profiled sample becomes a reviewable/editable schema through the *same* `PATCH /datasets/{id}` endpoint C0 already has. No new review UI concept needed. |
| From-description path | Completely untouched: `spec.source != "sample"` always routes through the existing `TabularSampler` + `remote_generate_shard` (byte-identical code path to C0). |

## 1. Domain additions — `anodyne-dataset` (additive only)

- `ColumnProfile` — name, semantic_type, nullable, null_rate, numeric stats (min/max/mean/std),
  categorical stats (`categories: dict[str, float]` — value→relative frequency, top-K), distinct_count.
- `Profile` — id, tenant_id, dataset_id, row_count, `columns: list[ColumnProfile]`,
  `correlations: dict[str, dict[str, float]]` (Pearson, numeric columns only), `sample_uri`,
  `sample_filename` (needed to re-read the raw sample's format later), created_at.
- `SampleProfiler` port — `profile(tenant_id, dataset_id, sample_uri, data: bytes, filename: str) -> Profile`
  (sync — CPU-bound, mirrors `Generator.generate`, run via `asyncio.to_thread` from async callers).
- `ProfileRepository` port — **new, separate ABC** (not added to `DatasetRepository`) so every
  existing `DatasetRepository` fake across the test suite keeps working unmodified:
  `save_profile(profile) -> None`, `get_profile(tenant_id, dataset_id) -> Profile | None`.
  `SqlDatasetRepository` implements both ABCs (multiple inheritance) since one SQL repository
  naturally serves both; nothing else needs to change.

## 2. `anodyne-tabular` — the new package

- `io.py` — `read_sample(data, filename) -> pandas.DataFrame` (csv/parquet by extension;
  `UnsupportedSampleFormatError` otherwise).
- `profiler.py` — `PandasSampleProfiler(SampleProfiler)`: dtype + name/regex heuristics
  (email/name/address by column name; datetime dtype; bool; int vs float; object columns with
  ≤ `max_categories` (default 50) distinct values → categorical, else → text), null-rate, numeric
  stats, top-K category frequencies, Pearson correlation over numeric columns. Caps profiling to
  `max_profile_rows` (default 200k, random-sampled) for large uploads.
- `schema.py` — `fields_from_profile(profile) -> list[FieldSpec]`.
- `constraints.py` — `enforce(table, fields) -> pyarrow.Table`.
- `realistic.py` — `faker_column(field, count, rng_seed) -> pyarrow.Array` (Faker, honoring
  `constraints["faker_locale"]`/`constraints["provider"]="mimesis"`), used for PII-like fields by
  every synthesizer below.
- `copula_generator.py` — `CopulaTabularGenerator(Generator)`: fits `rdt.HyperTransformer` on the
  modeled columns (numeric/boolean/categorical/datetime — i.e. everything **except** name/email/
  address/text), fits `copulas.multivariate.GaussianMultivariate` on the transformed numeric frame;
  `generate()` reseeds, samples, `HyperTransformer.reverse_transform`s, assembles PII/text columns
  via `realistic.py`, reorders to `spec.fields`, runs `constraints.enforce`.
- `deep_generator.py` — `DeepTabularGenerator(Generator)`: wraps `ctgan.CTGAN`/`ctgan.TVAE` directly
  on the modeled columns (`discrete_columns` = boolean/categorical), same reseed/assemble/enforce
  pipeline. `kind: Literal["ctgan","tvae"]`, `epochs` configurable (small in tests).
- `sdv_adapter.py` — `SdvGaussianCopulaGenerator(Generator)`, deferred `sdv` import,
  `SdvNotEnabledError` if `enabled=False`.
- `builder.py` — `build_tabular_generator(method, profile, sample, *, epochs=100, enable_sdv=False) -> Generator`
  dispatch (`"copula"` default / `"ctgan"` / `"tvae"` / `"sdv"`; unknown method → clear `ValueError`).

## 3. Orchestration extensions

- `anodyne_workflows.workflow.GenerationInput` gains `method: str = "copula"` (new field, default
  at the end → every existing keyword-argument call site in C0 is unaffected).
- `anodyne_workflows.activities.ActivityContext` gains two optional fields, both defaulting to
  `None`/safe values so every existing `ActivityContext(repo=..., s3_bucket=..., s3_client=...)`
  construction in C0's tests keeps working: `profile_repo: ProfileRepository | None = None`,
  `ctgan_epochs: int = 100`, `enable_sdv: bool = False`.
- `generate_shards` activity: unchanged for `spec.source != "sample"` (exact C0 code path,
  byte-for-byte). New branch for `spec.source == "sample"`: fetch the `Profile` (404-equivalent
  `ValueError` if missing — sample never uploaded), fetch the raw sample bytes from the object
  store, build the generator once (`asyncio.to_thread(build_tabular_generator, ...)`), then dispatch
  one `anodyne_compute.sample_tasks.remote_generate_shard_from_generator.remote(...)` Ray task per
  shard (shipping the fitted generator, not refitting).
- `anodyne_compute.sample_tasks` — new module (existing `ray_tasks.py` untouched):
  `generate_shard_bytes_from_generator` / `remote_generate_shard_from_generator`.

## 4. Storage — `anodyne-storage`

New tenant-scoped table `dataset_profiles` (primary key = `dataset_id`, one profile per dataset —
re-upload replaces it via upsert): `tenant_id`, `row_count`, `columns` (JSONB), `correlations`
(JSONB), `sample_uri`, `sample_filename`, `created_at`. Migration `0003_dataset_profiles.py`
(mirrors `0002`). `SqlDatasetRepository` additionally implements `ProfileRepository`.

## 5. Gateway API — `apps/api-gateway`

- `CreateDatasetRequest` gains `source: str = "description"` (default preserves all C0 behavior)
  and `description`/`target_rows` become optional-with-defaults for the sample path (schema/row
  count come from the profiled sample instead).
- `POST /datasets` — when `source == "sample"`, skips the LLM proposer entirely, creates a draft
  spec with empty `fields` (`datasets:write`).
- `POST /datasets/{id}/sample` (multipart upload, `datasets:write`) — validates
  `spec.source == "sample"`, size-caps the upload (25 MB), stores raw bytes to the object store,
  profiles it, persists the `Profile`, sets `spec.fields = fields_from_profile(profile)` and
  `spec.target_rows` (if unset) from `profile.row_count`, returns `{dataset, profile}`.
- `POST /datasets/{id}/generate` — resolves `GenerationInput.method` from
  `spec.directives.get("synthesizer", "copula")`; unchanged otherwise.
- New DI: `get_profile_repo`, `get_sample_profiler` (both overridable in tests, same pattern as
  every other gateway dependency).

## 6. Testing strategy

- **Unit:** profiler semantic-type inference + stats correctness; `fields_from_profile`;
  `constraints.enforce` (clipping/choice-restriction); copula generator determinism
  (same seed+range ⇒ identical bytes) on a small synthetic sample; gateway route tests (fakes,
  RBAC, 400s on wrong `source`, oversized upload).
- **Integration (marked `integration`, Docker/slow):** `dataset_profiles` RLS isolation (testcontainers);
  Ray shard dispatch of a fitted generator (local Ray); CTGAN/TVAE fit+sample on a tiny synthetic
  frame with `epochs=1`; SDV adapter (`pytest.importorskip("sdv")`, skipped when the optional extra
  isn't installed) verifying the `enabled=False` guard and, if installed, a fit+sample smoke test.

## Non-goals (C1)

Web UI for sample upload (deferred to a later UI pass); automatic synthesizer selection heuristics
(user/operator picks via `directives`); multi-table/relational synthesis; differential-privacy
guarantees beyond constraint enforcement; GPU-accelerated CTGAN (CPU-only, small epochs assumed for
CI; production tuning is a follow-up).
