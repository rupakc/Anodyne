# Plan — Sub-system D — Perturbation Module

TDD throughout: failing test first, then implement. Commit after each green group.

## 1. Domain (anodyne-dataset)
- Tests: `test_perturb_models.py`, `test_perturb_ports.py`.
- Add `PerturbationFamily`, `PerturbationSpec`, `PerturbationJob` to `models.py`.
- Add `parent_version_id` to `DatasetVersion`.
- Add `Perturbator` + `PerturbationRepository` ports to `ports.py`.

## 2. Adapter package (anodyne-perturbation)
- Scaffold package (pyproject src-layout) + register in root `pyproject.toml` dev group
  + `[tool.uv.sources]`.
- Tests: `test_perturb_params.py`, `test_perturb_noise.py`, `test_perturb_drift.py`,
  `test_perturb_outliers.py`, `test_perturb_bias.py`, `test_perturb_edgecase.py`,
  `test_perturb_text.py`, `test_perturb_registry.py`, `test_perturb_determinism.py`.
- Implement `params.py`, `tabular.py`, `text.py`, `registry.py`, `handlers.py`,
  `perturbator.py`.

## 3. Storage
- Add `perturbation_jobs` table + `dataset_versions.parent_version_id` to `db.py`;
  register table in `_TENANT_TABLES`.
- Extend `SqlDatasetRepository` with `PerturbationRepository` methods + parent_version_id
  round-trip in add_version/_version_from_row.
- Migration `perturbation_jobs.py` (`revision="perturbation_jobs"`, down `"0006"`).
- Integration test `test_perturbation_jobs_repo.py` (@integration) mirroring RLS tests.

## 4. Workflow
- Tests: `test_perturbation_workflow.py` (@integration), `test_perturbation_activities.py`.
- `perturbation_workflow.py` (`PerturbationWorkflow`, `PerturbationInput`).
- `perturbation_activities.py` (context + 3 activities + status), self-import handlers.
- Add `anodyne-perturbation` to workflows deps.

## 5. Worker wiring
- Register workflow + activities in `generation_worker.main` (build_worker), bind context.
- Test: extend/add worker wiring assertions (`test_perturbation_worker_wiring.py`).

## 6. API
- `perturbation_routes.py` APIRouter; include in `create_app`; `get_perturbation_repo` dep.
- Add `perturbations:read/write` to `RoleBasedPolicy`.
- Test: `test_perturbation_routes.py`.

## 7. Verify
- `uv run ruff check .`, `uv run mypy .`, `uv run pytest -m "not integration"`.
- Self-review diff; report.
