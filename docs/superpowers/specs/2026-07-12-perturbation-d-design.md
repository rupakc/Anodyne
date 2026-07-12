# Sub-system D — Perturbation Module (design)

Requirements 3 & 4: inject controlled **noise, drift, outliers/anomalies, bias, and
edge-cases** into an already-generated dataset for robustness / bias testing.

## Goal

Take a stored `DatasetVersion` and emit a **new derived `DatasetVersion`** whose rows
carry controlled, deterministic corruption. The derived version records its lineage
(`parent_version_id`) so a UI can show "v2 = Gaussian-noise(v1)".

## Where things live (hexagonal)

Mirrors the Generation Engine exactly.

- **Domain models + port** live in `anodyne-dataset` (the same layer as `Generator`),
  NOT in an adapter package. `anodyne-core` is untouched (it holds only the LLM/auth
  primitives; the dataset domain lives in `anodyne-dataset`, where `Generator`,
  `DatasetSpec`, `DatasetVersion` already are).
  - `PerturbationFamily` (StrEnum): `noise | drift | outliers | bias | edge_case`.
  - `PerturbationSpec` (Pydantic): `family`, `intensity: float` (0..1), `target_fields:
    list[str]` (empty = all eligible), `params: dict[str, object]`. `params` is a dict —
    consistent with `FieldSpec.constraints` / `DatasetSpec.directives`; the adapter parses
    it into typed per-family param models (`params.py`).
  - `PerturbationJob` (Pydantic): mirrors `GenerationJob` + `parent_version_id`,
    `result_version_id`, and the embedded `PerturbationSpec`.
  - `Perturbator` (ABC port): `perturb(spec, table, modality, seed) -> pyarrow.Table`.
    Sync + seeded, exactly like `Generator.generate`.
  - `PerturbationRepository` (ABC port): save/get/list jobs. Kept separate from
    `DatasetRepository` (like `ProfileRepository`) so adding it breaks no existing fake.
  - `DatasetVersion` gains an optional `parent_version_id: UUID | None` (additive,
    defaults `None`, so every existing caller/test is unaffected).

- **Adapter** package `packages/anodyne-perturbation/` (`anodyne_perturbation`):
  - `params.py` — typed Pydantic param models per family + `parse_params`.
  - `tabular.py` — `perturb_tabular(spec, table, seed)`: all five families over
    numeric/categorical/string arrow columns.
  - `text.py` — `perturb_text(spec, table, seed)`: char/word typos, masking, plus the
    drift/bias/edge families over the string columns of a text artifact.
  - `registry.py` — the **modality registry**, a structural mirror of
    `anodyne_workflows.modality`: `PerturbationHandler` Protocol, `_REGISTRY`,
    `register_perturbation`, `get_perturbation_handler`, `registered_perturbation_modalities`.
  - `handlers.py` — `TabularPerturbationHandler`, `TextPerturbationHandler`, and
    `_UnsupportedModalityHandler` instances registered for `image/audio/video` (a clean
    seam that raises `NotImplementedError` — no fake media logic).
  - `perturbator.py` — `RegistryPerturbator(Perturbator)`: dispatches on `modality` via
    the registry. This is the single dispatch site, exactly like generation's `get_handler`.

## Dispatch decision

The generation registry keys on **modality**; we mirror that. A `PerturbationHandler`
owns "how to perturb this modality's artifact"; inside it dispatches on `spec.family`.
This keeps ONE dispatch style (the modality registry) and gives image/audio/video a clean
registered seam. Families are the per-modality behaviour, not a second registry.

## Families (tabular)

Deterministic via `np.random.default_rng([seed, family_ord, field_ord])` — the repo idiom.

- **noise**: numeric → additive Gaussian (`sigma = intensity * col_std`) or uniform;
  categorical → random swap to another category at rate `intensity`; string → delegates to
  text typos.
- **drift**: covariate (`x*scale + shift`), concept (relabel a target field's values),
  temporal (monotone trend over row order).
- **outliers**: point (push a fraction `intensity` of numeric cells to `mean ± k*std`);
  contextual (inject rare category combinations).
- **bias**: class imbalance (resample rows toward a target ratio on a class field);
  demographic skew (over-represent a chosen value of a chosen field).
- **edge_case**: boundary values (min/max of a column), nulls/empties (null a fraction of
  cells), format violations (type-valid but extreme: empty/whitespace strings, zeros).

## Families (text)

Operate on string columns: char typos (swap/delete/insert), word typos, word masking
(`[MASK]`), plus drift(concept relabel)/bias(resample)/edge_case(empty/whitespace) reusing
the row-level ops. No network — pure string ops seeded by the same RNG idiom.

## Workflow (durable job)

New, separate from generation (keeps `workflow.py` untouched & import-free of adapters):

- `anodyne_workflows.perturbation_workflow.PerturbationWorkflow` + `PerturbationInput`.
  Orchestration only, dispatches activities by **name**.
- `anodyne_workflows.perturbation_activities`:
  - `set_perturbation_status` (mirrors `set_status`: upsert job + Redis publish).
  - `apply_perturbation`: load parent artifact → `RegistryPerturbator` (via `to_thread`) →
    upload derived artifact → return `(uri, rows)`.
  - `register_perturbed_version`: `add_version(DatasetVersion(..., parent_version_id=...))`
    and stamp `result_version_id` on the job. Matches generation's `register_version`.
- The activities self-import `anodyne_perturbation.handlers` for its registration side
  effect (exactly like `activities.py` imports `handlers`).
- Worker (`generation-worker`) registers the new workflow + activities alongside the
  generation ones on the same task queue; `PerturbationActivityContext` bound in
  `configure_perturbation_activities`.

Perturbation runs in-activity (via `asyncio.to_thread`), not on Ray — a whole-artifact
transform is a single CPU step; a Ray shard-map is a documented future seam.

## DB

- `dataset_versions` gains `parent_version_id UUID NULL` (lineage).
- `perturbation_jobs` table: id, tenant_id, dataset_id, parent_version_id, family, params
  JSONB, intensity, target_fields JSONB, status, progress, message, workflow_id,
  result_version_id NULL, created_at. RLS on `tenant_id` like every other tenant table.
- Migration file **`perturbation_jobs.py`** (descriptive, not `0007_*`),
  `revision = "perturbation_jobs"`, `down_revision = "0006"`.

## API (focused module `api_gateway/perturbation_routes.py`, an `APIRouter`)

- `POST /datasets/{dataset_id}/versions/{version_id}/perturb` (202) — validate version
  ownership, create a `PerturbationJob`, `start_workflow(PerturbationWorkflow.run, ...)`,
  save job, return it.
- `GET /perturbation-jobs/{job_id}` — tenant-scoped get.
- `GET /datasets/{dataset_id}/perturbation-jobs` — list for a dataset.
- New permissions `perturbations:read` / `perturbations:write` added to `RoleBasedPolicy`.

## Determinism contract

`(seed, spec)` ⇒ identical output; different `seed` ⇒ different output (tested). Each
family has a behaviour test on a small fixture. Text tests are pure/offline.

## Test layout

Globally-unique basenames, no `tests/__init__.py`, importlib mode. Docker/RLS tests
marked `@pytest.mark.integration`; default suite offline.
