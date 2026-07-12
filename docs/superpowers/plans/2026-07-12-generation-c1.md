# Generation C1 — Tabular (Full / From-Sample) Implementation Plan

**Goal:** From-sample tabular generation: profile an uploaded CSV/Parquet sample, synthesize with a
permissive statistical/deep stack (copulas+rdt default, CTGAN/TVAE opt-in, SDV opt-in adapter),
enforce constraints, Ray-shard at scale, wire through the existing Temporal `GenerationWorkflow`.
From-description generation (`TabularSampler`) is untouched.

**Architecture:** New package `anodyne-tabular`. Additive-only changes to `anodyne-dataset`
(2 models, 2 ports — `ProfileRepository` is a *separate* ABC so no existing `DatasetRepository` fake
needs updating). Additive changes to `anodyne-workflows` (`GenerationInput.method`, optional
`ActivityContext` fields, a new branch in `generate_shards`), `anodyne-compute` (new
`sample_tasks.py`, `ray_tasks.py` untouched), `anodyne-storage` (new table + migration + extra
interface on `SqlDatasetRepository`), `apps/api-gateway` (new route + optional request field),
`apps/generation-worker` (wire the two new deps). See the design spec for full rationale.

## Global constraints (same as C0 — do not relitigate)

- Python 3.12+, uv workspace. Register every new package in root `pyproject.toml`
  (`[dependency-groups] dev` + `[tool.uv.sources]`); `uv sync` after each new package.
- `ruff` + `mypy --strict` clean; `uv run pytest -q -m "not integration and not e2e"` green after
  every task.
- Globally-unique test basenames (prefix `test_tabular_*` for the new package; extend existing
  files in place elsewhere — do not create a second `test_activities.py` etc). No `tests/__init__.py`.
- Docker/Ray/Temporal/slow-training tests marked `integration`.
- Deterministic given a seed (see spec §"Determinism").
- Do **not** edit `packages/anodyne-generation/src/anodyne_generation/{sampler,proposer}.py` —
  import from them if ever needed (not expected in this plan).
- `sdv` is never added to root `dependency-groups.dev`/`uv.lock` — it is an optional extra on
  `anodyne-tabular` only (`anodyne-tabular[sdv]`), installed ad hoc for its own test if present.
- Commit per task, conventional commits.

---

### Task 1: `anodyne-dataset` — `Profile`/`ColumnProfile` models + `SampleProfiler`/`ProfileRepository` ports

**Files:** modify `packages/anodyne-dataset/src/anodyne_dataset/{models,ports}.py`;
extend `packages/anodyne-dataset/tests/{test_dataset_models.py,test_ports.py}`.

- [ ] Failing tests: `ColumnProfile` defaults (nullable=False, categories=None); `Profile` round-trips
  via `model_dump`/`model_validate`; a `class _FakeProfiler(SampleProfiler)` / `class _FakeProfileRepo(ProfileRepository)`
  can be instantiated once all abstract methods are implemented (signature-shape test, mirrors
  existing `test_ports.py` style).
- [ ] Implement `ColumnProfile`, `Profile` in `models.py`; `SampleProfiler`, `ProfileRepository` in
  `ports.py`. Confirm **no existing symbol is modified**, only appended.
- [ ] `uv run pytest packages/anodyne-dataset -q` green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(dataset): add sample profile models and profiler/profile-repo ports`.

---

### Task 2: `anodyne-tabular` — package skeleton + `io.py`

**Files:** create `packages/anodyne-tabular/pyproject.toml`, `src/anodyne_tabular/{__init__,io}.py`,
`tests/test_tabular_io.py`; modify root `pyproject.toml`.

- [ ] Failing tests: `read_sample(csv_bytes, "sample.csv")` returns a `pandas.DataFrame` with correct
  columns/values; same for `"sample.parquet"` (round-trip via `pyarrow`/`pandas.to_parquet`);
  unknown extension raises `UnsupportedSampleFormatError`.
- [ ] `pyproject.toml` deps: `["anodyne-core","anodyne-dataset","pandas>=2.2","pyarrow>=17"]` +
  workspace sources. Implement `io.py`.
- [ ] Register package in root `pyproject.toml` (`dependency-groups.dev` + `tool.uv.sources`);
  `uv sync`.
- [ ] `uv run pytest packages/anodyne-tabular -q` green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): scaffold anodyne-tabular package with sample reader`.

---

### Task 3: `anodyne-tabular` — `PandasSampleProfiler`

**Files:** create `src/anodyne_tabular/profiler.py`; `tests/test_tabular_profiler.py`.

- [ ] Failing tests on a small synthetic `pandas.DataFrame` (age: int; score: float with a null;
  is_active: bool; plan: 2-value categorical; email: emails; signup_at: datetime strings):
  - integer/float columns get correct min/max/mean/std; `score`'s null produces `null_rate > 0`.
  - `is_active` → `SemanticType.BOOLEAN`.
  - `plan` (≤ `max_categories`) → `CATEGORICAL` with `categories` frequencies summing to ~1.0.
  - column named `email` with `@` values → `SemanticType.EMAIL`.
  - `signup_at` (datetime dtype) → `SemanticType.DATETIME`.
  - `correlations` includes an entry for the numeric columns and is symmetric
    (`corr["age"]["score"] == corr["score"]["age"]`).
  - `profile.row_count == len(df)`, `profile.sample_uri`/`sample_filename` echoed from args.
  - a > `max_profile_rows` frame is subsampled (row_count reflects the sampled size, not the input —
    document this explicitly in the test name/assert).
- [ ] Implement `PandasSampleProfiler(SampleProfiler)` per the spec's heuristics.
- [ ] Tests green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): add pandas-based sample profiler`.

---

### Task 4: `anodyne-tabular` — `fields_from_profile` + `constraints.enforce`

**Files:** create `src/anodyne_tabular/{schema,constraints}.py`;
`tests/test_tabular_schema.py`, `tests/test_tabular_constraints.py`.

- [ ] Failing tests: `fields_from_profile` maps each `ColumnProfile` → `FieldSpec` with matching
  `semantic_type`/`nullable`, numeric `constraints={"min":...,"max":...}`, categorical
  `constraints={"choices":[...]}` (from `categories` keys). `enforce()`: numeric values outside
  min/max are clipped; categorical values outside the choice set are replaced with the most
  frequent declared choice; column order/name matches `spec.fields` exactly.
- [ ] Implement both modules.
- [ ] Tests green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): add profile-to-schema mapping and constraint enforcement`.

---

### Task 5: `anodyne-tabular` — `realistic.py` (Faker/Mimesis for PII-like fields)

**Files:** create `src/anodyne_tabular/realistic.py`; `tests/test_tabular_realistic.py`.
Add `mimesis` to the package's deps.

- [ ] Failing tests: `faker_column` for `NAME`/`EMAIL`/`ADDRESS`/`TEXT` produces `count` plausible
  values; same `(field, count, rng_seed)` ⇒ identical output (determinism); `constraints={"provider":
  "mimesis"}` produces values via Mimesis instead of Faker (assert via a distinguishing shape, e.g.
  Mimesis's default email domains differ from Faker's, or monkeypatch/spy both providers to confirm
  which is called).
- [ ] Implement `faker_column`.
- [ ] Tests green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): add deterministic realistic-value generation for PII-like fields`.

---

### Task 6: `anodyne-tabular` — `CopulaTabularGenerator` (default synthesizer)

**Files:** create `src/anodyne_tabular/copula_generator.py`; `tests/test_tabular_copula_generator.py`.
Add `copulas`, `rdt` to deps.

- [ ] Failing tests on a small synthetic sample (~200 rows, numeric+categorical+boolean+one email
  column): `generate(spec, 0, 50, seed=7)` twice with the same args ⇒ identical `pyarrow.Table`
  (determinism); output row count/column names match `spec.fields`; numeric columns stay within the
  sample's observed min/max (constraint enforcement engaged); the email column is Faker-generated
  (not copied from the input sample — assert no output value equals an input sample value, or at
  least that it's plausible-email-shaped and doesn't match input rows 1:1); disjoint shards
  (`start_row=0` vs `start_row=50`) produce different rows.
- [ ] Implement `CopulaTabularGenerator` (rdt `HyperTransformer` fit/transform/reverse_transform +
  `copulas.multivariate.GaussianMultivariate` fit/sample + reseed-per-shard + `realistic.py` for
  PII columns + `constraints.enforce`).
- [ ] Tests green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): add copula-based default from-sample generator`.

---

### Task 7: `anodyne-tabular` — `DeepTabularGenerator` (CTGAN/TVAE, opt-in)

**Files:** create `src/anodyne_tabular/deep_generator.py`; `tests/test_tabular_deep_generator.py`
(marked `integration` — real (tiny) training). Add `ctgan` to deps.

- [ ] Failing test (marked `integration`, `epochs=1`, ~100-row synthetic sample): fit+sample
  produces the right row count/columns for both `kind="ctgan"` and `kind="tvae"`; same seed ⇒
  identical output; unknown `kind` raises `ValueError` at construction (fast, no `integration` mark
  needed for that one assertion — split into its own non-integration test).
- [ ] Implement `DeepTabularGenerator`.
- [ ] `uv run pytest packages/anodyne-tabular -m integration -q` green (Docker not required — pure
  CPU/torch); non-integration lane still green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(tabular): add CTGAN/TVAE opt-in deep generator`.

---

### Task 8: `anodyne-tabular` — SDV opt-in adapter + `build_tabular_generator` dispatch

**Files:** create `src/anodyne_tabular/{sdv_adapter,builder}.py`;
`tests/test_tabular_sdv_adapter.py` (marked `integration`; `pytest.importorskip("sdv")`),
`tests/test_tabular_builder.py`. Add `[project.optional-dependencies] sdv = ["sdv>=1.15"]` to the
package's `pyproject.toml` (**not** referenced from root `dependency-groups.dev`).

- [ ] Failing tests: `build_tabular_generator("copula", ...)`/`"ctgan"`/`"tvae"` return the right
  class; `"sdv"` with `enable_sdv=False` raises `SdvNotEnabledError` (no `sdv` import needed for
  this assertion — keep it in the non-integration lane); unknown method raises `ValueError`.
  Integration-marked: with `sdv` installed and `enable_sdv=True`, fit+sample smoke test.
- [ ] Implement `sdv_adapter.py` (deferred `import sdv...` inside methods) and `builder.py`.
- [ ] Non-integration lane green without `sdv` installed (import-time safety is the point);
  `ruff`/`mypy --strict` clean (mypy: `sdv_adapter.py`'s deferred import needs
  `# type: ignore[import-not-found]` — acceptable, isolated to this one adapter).
- [ ] Commit: `feat(tabular): add opt-in SDV adapter and synthesizer dispatch`.

---

### Task 9: `anodyne-storage` — `dataset_profiles` table + migration + repository

**Files:** modify `src/anodyne_storage/db.py`; create migration `0003_dataset_profiles.py`;
modify `src/anodyne_storage/dataset_repo.py` (add `ProfileRepository` methods —
`SqlDatasetRepository(DatasetRepository, ProfileRepository)`); extend
`tests/test_dataset_repo.py` (marked `integration`, same testcontainers fixture already there).

- [ ] Failing integration test: `save_profile` + `get_profile` round-trip is tenant-isolated
  (another tenant's `get_profile` returns `None`); re-saving a profile for the same `dataset_id`
  replaces it (upsert, not a duplicate row).
- [ ] Add `dataset_profiles` table (PK = `dataset_id`) to `db.py` + `_TENANT_TABLES`; migration
  mirrors `0002` (`down_revision="0002"`); implement the two repository methods
  (`pg_insert(...).on_conflict_do_update(index_elements=[dataset_profiles.c.dataset_id], ...)`).
- [ ] `uv run pytest packages/anodyne-storage -m integration -q` green (Docker); non-integration
  lane unaffected; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(storage): add dataset_profiles table, migration, and repository methods`.

---

### Task 10: `anodyne-compute` — Ray dispatch of a pre-fit generator

**Files:** create `src/anodyne_compute/sample_tasks.py`; `tests/test_compute_sample_tasks.py`
(marked `integration` — local Ray). Modify `anodyne-compute/pyproject.toml` (dep on
`anodyne-tabular`).

- [ ] Failing test: given a tiny fitted `CopulaTabularGenerator` (from Task 6, built directly in the
  test on a synthetic sample — no Docker needed beyond local Ray), `remote_generate_shard_from_generator.remote(...)`
  and the local (non-Ray) `generate_shard_bytes_from_generator` produce byte-identical Parquet for
  the same args (mirrors the existing `test_ray_tasks.py::test_ray_remote_matches_local` pattern).
- [ ] Implement `sample_tasks.py` (same shape as `ray_tasks.py`, taking a `Generator` instance
  instead of always constructing `TabularSampler`).
- [ ] `uv run pytest packages/anodyne-compute -m integration -q` green; non-integration lane
  unaffected; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(compute): add Ray dispatch for pre-fit sample-based generators`.

---

### Task 11: `anodyne-workflows` — wire the from-sample path into `generate_shards`

**Files:** modify `src/anodyne_workflows/{workflow,activities}.py`;
extend `tests/test_activities.py` (add cases; do not create a second file).

- [ ] Failing tests: `GenerationInput(...)` still constructs with all C0 call sites unchanged
  (regression — no `method` kwarg needed, defaults to `"copula"`); a `generate_shards`-level test
  with `spec.source == "sample"` and a fake `ProfileRepository`/object store returns shard keys
  produced via the sample path (assert the fake generator-builder was invoked once, not once per
  shard — this is the fit-once assertion); missing profile raises a clear error before touching Ray.
- [ ] Add `method: str = "copula"` to `GenerationInput`; add `profile_repo`, `ctgan_epochs`,
  `enable_sdv` (all defaulted) to `ActivityContext`; add the `spec.source == "sample"` branch in
  `generate_shards` per the spec (fit once via `asyncio.to_thread`, then one Ray task per shard).
- [ ] Full non-integration + integration suites green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(workflows): route sample-sourced datasets through the tabular synthesis stack`.

---

### Task 12: `apps/generation-worker` — wire the two new deps

**Files:** modify `src/generation_worker/{config,main}.py`; extend `tests/test_worker_wiring.py`.

- [ ] Failing test: `build_worker` still registers exactly the same 5 activities on `"generation"`
  (regression); `WorkerDeps`/`ActivityContext` wiring passes through `profile_repo`
  (`SqlDatasetRepository` reused — it implements both ABCs), `ctgan_epochs`, `enable_sdv` from new
  `Settings` fields (`tabular_ctgan_epochs: int = 100`, `tabular_enable_sdv: bool = False`).
- [ ] Implement.
- [ ] Tests green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(generation-worker): wire sample-profiling settings and repository`.

---

### Task 13: `apps/api-gateway` — sample upload endpoint + `source` on create + method resolution

**Files:** modify `src/api_gateway/{app,deps,config}.py`; create
`tests/test_dataset_sample_routes.py` (new file — do not touch `test_dataset_routes.py`'s existing
cases beyond what's needed for the `source` default regression, if any).

- [ ] Failing tests: `POST /datasets {source:"description", description:"x"}` behaves exactly as
  before (regression, in the new file or asserted still-passing in the existing one); `POST /datasets
  {source:"sample", name, target_rows:0}` returns 201 with empty `fields` and no LLM call;
  `POST /datasets/{id}/sample` with a small CSV upload returns 200 with a populated `fields` list and
  a `profile`, and a follow-up `GET /datasets/{id}` reflects the new fields; uploading to a
  `source="description"` dataset → 400; oversized upload → 413; `datasets:write` required (viewer
  → 403); `POST /datasets/{id}/generate` on a sample dataset threads `directives["synthesizer"]`
  into the `GenerationInput.method` sent to the (fake) Temporal client.
- [ ] Add `source` to `CreateDatasetRequest`; implement `POST /datasets/{id}/sample`; add
  `deps.get_profile_repo`/`deps.get_sample_profiler`; resolve `method` in `start_generation`.
- [ ] Full non-integration suite green; `ruff`/`mypy --strict` clean.
- [ ] Commit: `feat(gateway): add sample upload endpoint and from-sample dataset creation`.

---

### Task 14: Final pass — root registration, lock, full suite, docs

**Files:** root `pyproject.toml`/`uv.lock`; `README.md` roadmap line (if present, mark C1);
`docs/architecture.md` (no change expected — already accurate).

- [ ] `uv sync && uv run pytest -q -m "not integration and not e2e"` and
  `uv run pytest -q -m integration` (where Docker/local-Ray available) both green.
  `uv run ruff check . && uv run mypy .` clean repo-wide.
- [ ] Self-review: reread every new/changed file with fresh eyes; check for placeholders, dead
  code, and that determinism claims actually hold (rerun the copula/CTGAN determinism tests twice).
- [ ] Commit: `chore: finalize Generation C1 (tabular from-sample)`.
- [ ] Push `feat/generation-c1-tabular` to origin. Do not merge, do not touch other branches.

---

## Self-Review

**Spec coverage:** `SampleProfiler` → T1/T3 ✓; permissive statistical stack (copulas+rdt default)
→ T6 ✓; CTGAN/TVAE → T7 ✓; SDV opt-in → T8 ✓; Faker/Mimesis + constraints → T4/T5 ✓;
`source="sample"` gateway path → T13 ✓; Temporal/Ray extension → T10/T11/T12 ✓; from-description
untouched → asserted as regressions in T11/T13.

**Non-conflict guarantee:** no line of `packages/anodyne-generation/**` is touched by this plan;
`anodyne-dataset` gets append-only changes (verified by "no existing symbol modified" checks in
T1); every existing `DatasetRepository`/`ActivityContext`/`GenerationInput` construction across the
C0 test suite keeps compiling because new fields are optional-with-defaults and the new
`ProfileRepository` port is separate from `DatasetRepository`.

**Type/name consistency:** `Profile`/`ColumnProfile` (T1) flow unchanged through
`fields_from_profile`/`enforce`/every generator (T3–T8) into the repository (T9), the Ray task
(T10), the activity (T11), the worker (T12), and the gateway (T13) — same field names throughout
(`sample_uri`, `sample_filename`, `row_count`, `columns`, `correlations`).
