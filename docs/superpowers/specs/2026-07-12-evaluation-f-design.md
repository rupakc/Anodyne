# Sub-system F — Evaluation Engine (LLM-as-a-Judge, Mixture-of-Experts, 360°)

> Requirements 7, 8, 9. Spec → plan → TDD. Branch `feat/evaluation-f`.

## 1. Goal

Evaluate a `DatasetVersion` (optionally synthetic-vs-reference) with a **mixture of expert
judges**, combine their verdicts into a weighted **360° `EvaluationReport`**, and persist a
scored **report artifact** (JSON + self-contained HTML) tenant-prefixed in the object store.
Runs as a Temporal workflow whose single work activity fans the statistical experts out over
Ray; the qualitative expert calls the tenant's LLM strictly through the `LLMProvider` port.

## 2. Where things live (hexagonal, matches repo)

New package `packages/anodyne-evaluation/` is a **bounded context** modelled on `anodyne-video`
(own `models.py` + `ports.py` + adapter subpackage). It is an *adapter* package: it may import
`anodyne-storage`, `anodyne-tabular`, `anodyne-llm`, but **`anodyne-core` imports nothing new**
(hard constraint upheld — core is untouched).

- `anodyne_evaluation/models.py` — domain models: `EvalDimension`, `ExpertScore`,
  `EvaluationReport`, `EvaluationRun`, `EvaluationStatus`, `EvaluationConfig`.
- `anodyne_evaluation/ports.py` — `Judge` (ABC, the expert port), `Aggregator`,
  `EvaluationRepository`, `JudgeRunner` (async fan-out strategy), `EvaluationContext`
  (frozen dataclass carrying the in-memory DataFrames; pandas imported under `TYPE_CHECKING`
  so the port module stays light), and `JudgeNotApplicable`.
- `anodyne_evaluation/judges/` — the six experts (one file each) + `base.StatisticalJudge`.
- `anodyne_evaluation/aggregator.py` — `WeightedAggregator`.
- `anodyne_evaluation/report.py` — `render_json` + `render_html` (autumn-pastel, self-contained).
- `anodyne_evaluation/evaluator.py` — `MoEEvaluator` (orchestrates judges via a `JudgeRunner`).
- `anodyne_evaluation/registry.py` — `SqlEvaluationRepository` (imports `anodyne_storage.db`,
  mirrors `SqlDatasetRepository`: `tenant_session` + explicit `tenant_id` filter = RLS +
  defense-in-depth).
- `anodyne_evaluation/loader.py` — load a version artifact (Parquet bytes) → DataFrame.

Shared/touched: `anodyne_storage.db` (+ two tables + RLS entries), a new Alembic migration
`0007` (`down_revision="0006"`), `anodyne_compute/ray_evaluation.py` (Ray `JudgeRunner`),
`anodyne_workflows` (evaluation workflow + activities), `apps/evaluation-worker` (new worker),
`apps/api-gateway` (routes module + deps + authz permissions), root `pyproject.toml`.

## 3. The `Judge` port (mixture-of-experts)

```python
class Judge(ABC):
    dimension: EvalDimension
    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore: ...
```

`ExpertScore` = `dimension`, `score` (normalized **0..1, higher is better**), `rationale`,
`metrics` (raw numbers), `recommendations`. A judge raises `JudgeNotApplicable` when its
preconditions are unmet (e.g. fidelity/privacy/utility without a reference, bias without a
sensitive field); the aggregator then excludes it and renormalizes the remaining weights — this
is what makes the score a **360° view of whatever dimensions are measurable**.

`StatisticalJudge(Judge)` adds a synchronous `compute(ctx) -> ExpertScore`; its `evaluate` just
returns `compute`. This split is what lets the Ray runner dispatch the CPU-bound experts as
`@ray.remote` tasks while the qualitative (LLM, async) judge runs inline.

### Experts

1. **FidelityJudge** (needs reference; no LLM). Per numeric column KS statistic
   (`scipy.stats.ks_2samp`); per categorical column Jensen–Shannon distance
   (`scipy.spatial.distance.jensenshannon`) over aligned category frequencies; correlation-matrix
   delta = mean abs difference of the two numeric correlation matrices. Column semantic types are
   taken from `anodyne_tabular.PandasSampleProfiler` (**reuse of the profiling layer**).
   `score = 1 - mean([ks_mean, js_mean, corr_delta])`.
2. **DiversityJudge** (subject only; no LLM). Mean column uniqueness ratio, mean normalized
   Shannon entropy of categoricals, and mode-collapse = max single-category frequency.
   `score = mean(mean_uniqueness, mean_entropy) * (1 - max_mode_freq_penalty)`.
3. **PrivacyJudge** (needs reference; no LLM). Exact-duplicate/memorization rate (subject rows
   present verbatim in reference) + nearest-neighbour distance ratio (DCR) on standardized numeric
   columns via `sklearn.neighbors.NearestNeighbors`. `score = 1 - privacy_risk`.
4. **UtilityJudge** (TSTR; needs reference + `target_field`; no LLM). Train a small **seeded**
   sklearn model on synthetic, test on real (TSTR); compare to train-real/test-real (TRTR).
   Classification (categorical target) → accuracy; regression → R². `score = clamp(TSTR/TRTR)`.
5. **BiasJudge** (needs `sensitive_field`; no LLM). Group representation entropy + demographic
   parity difference and disparate-impact ratio of `target_field` outcomes across groups.
   `score = 1 - demographic_parity_diff` blended with representation balance.
6. **QualitativeJudge** (LLM-as-a-Judge; uses `LLMProvider` port). Samples rows (tabular → text
   lines; text corpus → documents), sends a structured rubric prompt (realism, coherence,
   task_fit on 1–5), parses JSON (fence-stripped like `LLMSchemaProposer`).
   `score = mean(realism, coherence, task_fit) / 5`. **Mocked in unit tests** (fake `LLMProvider`
   returning canned JSON — no network).

## 4. Aggregation formula (explicit)

Default weights (sum 1.0): fidelity 0.25, privacy 0.20, utility 0.20, diversity 0.15,
qualitative 0.10, bias 0.10. Overridable per run via `EvaluationConfig.weights`.

```
present = {d: w_d for each dimension d that produced an ExpertScore}
overall = sum(present[d] * score_d) / sum(present.values())      # weighted mean over present dims
```

Skipped experts drop out of both numerator and denominator (renormalization), so the overall is
always a weighted mean over the dimensions actually measured. `recommendations` = concatenation
of each expert's recommendations; `summary` = human-readable one-liner keyed on the overall band
(>=0.8 strong / >=0.6 acceptable / else needs work). Tested explicitly for weighting +
renormalization + banding.

## 5. Orchestration, persistence, API

- **Temporal**: `EvaluationWorkflow.run(EvaluationInput)` → `set_eval_status(running, .1)` →
  `run_evaluation` (load artifacts, fan judges out via the injected `JudgeRunner`, aggregate,
  render JSON+HTML, upload tenant-prefixed `evaluations/{run_id}/report.{json,html}`, persist run
  + per-expert rows) → `set_eval_status(succeeded, 1.0)`. Mirrors `GenerationWorkflow`'s
  set-status/register cadence and its modality-agnostic, activity-name-only dispatch.
- **Ray**: `anodyne_compute.ray_evaluation.RayJudgeRunner` dispatches each `StatisticalJudge` via
  `@ray.remote` and awaits the qualitative judge inline (a `JudgeRunner`; the sequential runner is
  the offline default and test path). Ray test is `@pytest.mark.integration` (matches the other
  compute ray tests).
- **DB**: `evaluation_runs` + `evaluation_expert_results`, both `tenant_id` + RLS policy, added to
  `anodyne_storage.db` metadata and `_TENANT_TABLES`; Alembic `0007` (`down_revision="0006"`)
  mirrors `0006`.
- **API** (`api_gateway/evaluation_routes.py`, an `APIRouter`, tenant-ownership enforced by
  resolving the version through the caller's `DatasetRepository`):
  - `POST /datasets/{dataset_id}/versions/{version_id}/evaluate` (202) → create run + start workflow.
  - `GET /evaluations/{run_id}` → status/progress/overall.
  - `GET /evaluations/{run_id}/report` → the JSON report.
  - `GET /evaluations/{run_id}/report/download` → presigned URL of the HTML artifact.
  - Permissions `evaluations:read` (viewer+) / `evaluations:write` (member+).

## 6. Evidently

**Reference only — not a dependency.** The drift (KS/JS), data-quality, and column-correlation
metrics are *inspired by* Evidently's report families but implemented directly on
numpy/scipy/scikit-learn. No `evidently` import anywhere.

## 7. Testing

TDD, `--import-mode=importlib`, globally-unique test basenames, no `tests/__init__.py`. Default
suite fully offline: each judge on small fixtures with known expected ranges; aggregator weighting
+ renormalization exact; qualitative judge with an injected fake `LLMProvider`; evaluator
end-to-end sequential; report render (JSON round-trips, HTML self-contained/no external URLs);
gateway routes with fake repos + overridden auth. Docker/Ray → `@pytest.mark.integration`
(SqlEvaluationRepository RLS round-trip; RayJudgeRunner).

## 8. Decisions (autonomous)

- Domain models + `Judge` port live in the new bounded-context package (like `anodyne-video`),
  not in `anodyne-core`; core stays adapter-free (constraint upheld literally).
- One `run_evaluation` activity (not per-expert Temporal activities): DataFrames aren't worth
  round-tripping through Temporal payloads; the *intra-activity* fan-out is Ray's job, matching
  "Ray owns compute, Temporal owns flow."
- Scores normalized 0..1 higher-better across all experts so one weighted mean is meaningful.
- `scipy` + `scikit-learn` added (pinned) to the evaluation package — both already resolved in the
  environment; justified by fidelity/privacy/utility math.
