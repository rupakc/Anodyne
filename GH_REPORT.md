# Track GH — Graph perturbations

**Status: DONE.** Commit range `80a2830..b1d703e` (3 commits on `feat/graph-gh-perturb`).

## Files
- `packages/anodyne-dataset/src/anodyne_dataset/models.py:86` — added `GRAPH_REWIRE`/`GRAPH_DROPOUT`/`GRAPH_ONTOLOGY_VIOLATION` to `PerturbationFamily`.
- `packages/anodyne-graph/src/anodyne_graph/perturb.py` — **new**. `perturb_graph(dataset, family, intensity, seed, params=None) -> GraphDataset`.
- `packages/anodyne-perturbation/src/anodyne_perturbation/registry.py:53` — graph seam: `GraphPerturbationHandler` Protocol, `_GRAPH_REGISTRY`, `register_graph_perturbation`/`get_graph_perturbation_handler`/`registered_graph_perturbation_modalities`.
- `.../handlers.py` — `GraphPerturbationHandler` registers `"graph"`, delegating to `anodyne_graph.perturb`.
- `.../perturbator.py` — `RegistryPerturbator.perturb_graph(spec, dataset, seed, modality="graph")`.
- `.../__init__.py`, `pyproject.toml` — exports + `anodyne-graph` dependency.
- `packages/anodyne-workflows/src/anodyne_workflows/perturbation_activities.py:90` — `_is_graph`; graph branch in `apply_perturbation`.
- Tests (new, unique basenames): `test_graph_perturb.py` (10), `test_perturb_graph_registry.py` (4), `test_perturbation_graph_activity.py` (3). Scoped `test_perturb_determinism.py` to columnar families.

## Graph branch in the activity
When `parent.format == "graph_json"` (or modality GRAPH), `apply_perturbation` loads via `from_json_bytes` → `GraphDataset`, perturbs through `get_graph_perturbation_handler(modality)`, re-serializes via `to_json_bytes`, and returns `[key, node_count]`. Columnar artifacts keep the pa.Table path unchanged. `register_perturbed_version` reuses `parent.format`, so the child version stays `graph_json`.

## Design notes
- Separate graph registry (not the pa.Table one) so graphs never pass through `pyarrow.Table` and `registered_perturbation_modalities()` is unchanged.
- Deterministic: `default_rng([seed, family_ord])` over **id-sorted** selection; output preserves input order → intensity 0 is byte-exact no-op. Input never mutated (deep copies).
- PII: violation injection skips `is_pii` properties; no PII is ever synthesized.

## Deviations
None. No migrations (enum-only).

## Verification (final, from worktree root)
- `pytest packages/anodyne-graph packages/anodyne-perturbation packages/anodyne-workflows`: **192 passed, 2 warnings** (~93s).
- `ruff check` (four packages): **All checks passed!**
- `mypy .`: **Success: no issues found in 319 source files**.
