# Plan — Sub-system F (Evaluation Engine)

Branch `feat/evaluation-f`. TDD throughout: failing test → implementation → green. Commit per task.

## Task 1 — Package skeleton + domain models + ports
- `packages/anodyne-evaluation/pyproject.toml` (deps: core, dataset, storage, tabular, llm,
  numpy, pandas, scipy>=1.13, scikit-learn>=1.5, pyarrow; workspace sources).
- `models.py`: `EvalDimension`, `ExpertScore`, `EvaluationReport`, `EvaluationStatus`,
  `EvaluationRun`, `EvaluationConfig`.
- `ports.py`: `EvaluationContext` (frozen dataclass), `JudgeNotApplicable`, `Judge`, `Aggregator`,
  `EvaluationRepository`, `JudgeRunner`.
- Tests: `test_evaluation_models.py` (defaults/serialization), `test_evaluation_ports.py`
  (context construction, judge subclass contract).

## Task 2 — Statistical judges (offline, seeded)
- `judges/base.py` `StatisticalJudge`; `judges/fidelity.py`, `diversity.py`, `privacy.py`,
  `utility.py`, `bias.py`.
- Tests (known-range fixtures): `test_judge_fidelity.py`, `test_judge_diversity.py`,
  `test_judge_privacy.py`, `test_judge_utility.py`, `test_judge_bias.py`. Assert identical
  distributions → high fidelity/privacy-risk; disjoint → the opposite; NotApplicable paths.

## Task 3 — Qualitative LLM judge (mocked)
- `judges/qualitative.py` (uses `LLMProvider` port + `ModelConfig`; fence-stripped JSON parse).
- `test_judge_qualitative.py` with a fake `LLMProvider` returning canned JSON; assert score,
  metrics, and no network.

## Task 4 — Aggregator + evaluator
- `aggregator.py` `WeightedAggregator`; `evaluator.py` `MoEEvaluator` + `sequential_runner`.
- `test_aggregator.py` (exact weighted mean, renormalization when experts skipped, banding),
  `test_evaluator.py` (end-to-end sequential over fake judges → report).

## Task 5 — Report artifact
- `report.py` `render_json` + `render_html` (autumn-pastel, fully inline, no external URLs).
- `test_evaluation_report.py`: JSON round-trips to `EvaluationReport`; HTML contains scores and
  has no `http`/`src=`/`link` external refs.

## Task 6 — Persistence
- Add `evaluation_runs` + `evaluation_expert_results` to `anodyne_storage.db` (+ `_TENANT_TABLES`).
- Alembic `0007_evaluation_runs.py` (`down_revision="0006"`).
- `registry.py` `SqlEvaluationRepository`. `test_evaluation_repo.py` (`@pytest.mark.integration`,
  mirrors `test_dataset_repo.py`: RLS isolation, run upsert, expert-result round-trip).

## Task 7 — Ray runner
- `anodyne_compute/ray_evaluation.py` `RayJudgeRunner` (`JudgeRunner`). Register in compute
  `__init__` + pyproject dep on anodyne-evaluation. `test_ray_evaluation.py`
  (`@pytest.mark.integration`, local ray).

## Task 8 — Temporal workflow + activities
- `anodyne_workflows/evaluation_workflow.py` (`EvaluationInput`, `EvaluationWorkflow`),
  `evaluation_activities.py` (`EvaluationActivityContext`, `configure_evaluation_activities`,
  `set_eval_status`, `run_evaluation`), `loader.py` in evaluation pkg. Add evaluation dep to
  workflows pyproject. `test_evaluation_workflow.py` (`@pytest.mark.integration`, time-skipping,
  mocked activities), `test_evaluation_activities.py` (offline: fakes + sequential runner).

## Task 9 — evaluation-worker app
- `apps/evaluation-worker/` mirroring `generation-worker` (`build_worker`, `WorkerDeps`, `main`,
  `config`). `test_evaluation_worker_wiring.py` (offline, fakes).

## Task 10 — API gateway
- `api_gateway/evaluation_routes.py` (`APIRouter`), `deps.get_evaluation_repo`,
  `EvaluationRepository` structural type, authz `evaluations:read/write`. Wire `include_router`.
- `test_evaluation_routes.py` (offline: fake repos, overridden auth, tenant-ownership 404).

## Task 11 — Root wiring, migration id, full verification
- Root `pyproject.toml`: add `anodyne-evaluation` + `evaluation-worker` to dev group + sources.
- `uv sync`; `uv run ruff check .`; `uv run mypy .`; `uv run pytest -m "not integration"`.
- Self-review diff; report counts + shared files.
