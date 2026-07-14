# Per-Modality, Per-Task-Class Standard Metrics — Design

**Date:** 2026-07-14
**Sub-system:** F (Evaluation / LLM-as-a-Judge), with thin edges into api-gateway and apps/web.
**Status:** Approved for planning.

## 1. Problem

The LLM-as-a-Judge mixture-of-experts (`anodyne-evaluation`) dispatches on **modality
only**: graph runs get the `GRAPH_*` judges, everything else gets the same generic
synthetic-data-quality mixture (fidelity / diversity / privacy / utility / bias /
qualitative). Those judges answer *"is this good synthetic data?"* — distributional
fidelity, privacy leakage, diversity — not *"is this dataset good for its task?"*

Every modality **and** every task-class (text classification, summarization, QA, chat;
tabular classification/regression; image/audio classification and generation; text-to-video;
graph multi-hop QA) has its own set of **standard metrics**. Apart from the existing
fairness/bias tests, the evaluator must also score datasets on those standard metrics, and
the user must be able to **choose which standard metrics** to run for a given task.

## 2. Scope

**In scope (this spec):** a task-class-aware standard-metrics layer across all six
modalities, at the depth reachable through the **existing text-only `LLMProvider` port**
plus intrinsic (non-LLM) statistics. User-selectable metrics per task.

**Explicitly deferred (own future spec — "multimodal perception judge"):** extending
`LLMProvider`/`Message` to carry image/audio/video bytes, and the metrics that need true
media perception — vision label-accuracy from pixels, segmentation IoU/Dice, CLIP-style
caption adherence, ASR-based audio metrics. This spec does **not** touch `anodyne-core`.

### 2.1 Grounding facts (verified against the codebase)

- `anodyne_core.models.Message.content` is `str` — the port is text-only. Image/audio/video
  bytes cannot reach the LLM today.
- Media artifacts are `manifest.json`, not columnar: **image** items carry
  `{item_index, object_key, prompt, label, mime_type}`; **audio** `{index, object_key, text,
  label, voice, format, duration_seconds}`; **video** `{index, prompt, duration_seconds,
  width, height, fps, seed, object_key, ...}` (no label).
- `anodyne_text.shapes.detect_shape(fields)` already infers a text task-shape
  (classification / qa / summarization / chat / generic) purely from field names.
- `ExpertScore.metrics: dict[str, float]` already persists arbitrary named metrics; the
  storage repo round-trips it. No storage/schema change is required to surface the numbers.
- The aggregator renormalizes weights over exactly the dimensions that produced a score, so
  adding one dimension — even as the only one present for a media run — stays well-defined.

## 3. Architecture

All new code lives in `anodyne-evaluation` except: one loader helper, one activity edit
(`anodyne-workflows`), one API route + request field (`api-gateway`), and one UI panel
(`apps/web`). Hexagonal boundary preserved: `anodyne-core` untouched; the LLM is reached
only through the `LLMProvider` port.

### 3.1 Task-class detection — `anodyne_evaluation/task.py`

```python
class TaskType(StrEnum):
    TEXT_CLASSIFICATION = "text_classification"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    TABULAR_CLASSIFICATION = "tabular_classification"
    REGRESSION = "regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    AUDIO_CLASSIFICATION = "audio_classification"
    SPEECH_SYNTHESIS = "speech_synthesis"
    TEXT_TO_VIDEO = "text_to_video"
    GRAPH_QA = "graph_qa"
    GENERIC = "generic"

def detect_task(
    modality: Modality,
    columns: list[str],
    *,
    target_field: str | None = None,
    target_is_numeric: bool = False,
) -> TaskType: ...
```

Rules:
- **TEXT** — map `detect_shape` output: CLASSIFICATION→`TEXT_CLASSIFICATION`, QA→`QA`,
  SUMMARIZATION→`SUMMARIZATION`, CHAT→`CHAT`, GENERIC→`GENERIC`. Detection here works from
  column names (build throwaway `FieldSpec`s or replicate the subset check); evaluation may
  import `anodyne_text.shapes` (acyclic sibling dependency).
- **TABULAR** — `target_field` set & numeric → `REGRESSION`; set & non-numeric →
  `TABULAR_CLASSIFICATION`; unset → `GENERIC`.
- **IMAGE** — `"label"` column present → `IMAGE_CLASSIFICATION`, else `IMAGE_GENERATION`.
- **AUDIO** — `"label"` present → `AUDIO_CLASSIFICATION`, else `SPEECH_SYNTHESIS`.
- **VIDEO** → `TEXT_TO_VIDEO`.
- **GRAPH** — `GRAPH_QA` when the eval config supplies a QA-fixture set; otherwise the run
  uses only the existing GD judges and the task-metrics judge drops out (`GENERIC` →
  `JudgeNotApplicable`).

Override: `EvaluationConfig.task_type: str | None` wins when set (validated against
`TaskType`); otherwise `detect_task` runs.

### 3.2 Metric catalog + selection

```python
class MetricSpec(BaseModel):
    key: str          # stable id, e.g. "accuracy", "macro_f1", "faithfulness"
    label: str        # human label for the UI
    description: str
    requires_llm: bool  # informational; the whole judge already requires an LLM

class TaskMetricProvider(Protocol):
    task_type: TaskType
    def metric_catalog(self) -> list[MetricSpec]: ...
    async def score(
        self, ctx: EvaluationContext, provider: LLMProvider, model_config: ModelConfig,
        *, selected: frozenset[str],
    ) -> ExpertScore: ...
```

- A module-level `TASK_METRIC_PROVIDERS: dict[TaskType, TaskMetricProvider]` registry, plus
  `catalog_for(task_type) -> list[MetricSpec]`.
- Selection: `EvaluationConfig.selected_metrics: list[str] | None` (subset of catalog keys;
  `None` = all). `EvaluationContext.selected_metrics: frozenset[str] | None` carries it.
- The provider computes/aggregates **only** the selected metrics; the dimension `score` is
  the mean of the selected metrics' **0..1 contributions** (each provider maps its raw metric
  to a 0..1 contribution — e.g. accuracy is already 0..1; compression ratio maps through a
  target band). Selecting an empty or all-invalid set → `JudgeNotApplicable`.

### 3.3 The judge — `anodyne_evaluation/judges/task_metrics/judge.py`

`TaskMetricsJudge(provider, model_config)`, `dimension = EvalDimension.TASK_QUALITY`.
Appended to the mixture **only when an `LLMProvider` + `ModelConfig` are configured** (mirrors
`QualitativeJudge`), which is how "require an LLM" is enforced — no LLM ⇒ no dimension.
`evaluate` reads `ctx.task_type`, resolves the provider, resolves `selected` (config subset ∩
catalog, or full catalog when `None`), and delegates to `provider.score`. Unknown task,
unmet precondition (e.g. summarization run whose frame lacks `summary`), or empty selection
⇒ `JudgeNotApplicable`.

### 3.4 Metric providers (one module each, `judges/task_metrics/`)

Every provider: intrinsic metrics from `ctx.subject` (always) + a **sampled, `temperature=0`
LLM-oracle** pass through the port. Sample size = `ctx.sample_rows`. Determinism: sampling is
seeded by `ctx.seed`; prompts are fixed; JSON parsed with the same fence-strip as
`QualitativeJudge`.

| Provider | Catalog keys (metrics) |
|---|---|
| `text_classification` | `accuracy`, `macro_f1`, `per_class_precision`/`per_class_recall` (folded into macro), `class_balance`, `duplicate_rate` |
| `qa` | `answer_correctness`, `groundedness`, `answerable_rate`, `question_type_diversity` |
| `summarization` | `faithfulness`, `coverage`, `conciseness`, `compression_ratio`, `abstractiveness` |
| `chat` | `instruction_following`, `coherence`, `turn_validity` |
| `tabular_classification` | `label_consistency`, `class_balance`, `feature_completeness` |
| `regression` | `target_range_validity`, `target_distribution_health`, `feature_completeness` |
| `image` (both image tasks) | `label_balance` (cls), `prompt_label_consistency` (cls, LLM), `prompt_diversity`, `duplicate_rate` |
| `audio` (both audio tasks) | `label_balance` (cls), `transcript_label_consistency` (cls, LLM), `duration_uniformity`, `transcript_quality` (LLM) |
| `video` | `duration_conformance`, `resolution_consistency`, `fps_consistency`, `prompt_diversity`, `prompt_quality` (LLM) |
| `graph_qa` | `answer_groundedness`, `multi_hop_correctness`, `answerable_rate` |
| `generic` | `realism`, `coherence`, `task_fit` (a light rubric; the safety net) |

LLM-oracle definitions (all returned as JSON, mapped to 0..1):
- **classification accuracy/F1**: LLM predicts the label for each sampled row from its
  input text/features; compare to the stored label → accuracy and macro-F1 over the sampled
  confusion counts.
- **QA correctness/groundedness**: LLM judges whether the stored answer is correct for the
  question and supported by the context (when a context/`document` column exists).
- **summarization faithfulness/coverage/conciseness**: 1–5 rubric vs. the source document.
- **consistency** (image/audio): LLM judges whether the `prompt`/`text` plausibly describes
  the stated `label` — text-only, no media perception.

Intrinsic definitions: `class_balance` = normalized entropy of label counts;
`duplicate_rate` = 1 − (unique/total) on the primary field; `compression_ratio` =
mean(len(summary)/len(document)); `abstractiveness` = 1 − mean n-gram overlap(summary,
document); `duration/resolution/fps` conformance/consistency = fraction within tolerance of
the modal/requested value; `feature_completeness` = 1 − mean null-rate.

### 3.5 Plumbing

- **Loader** (`anodyne_evaluation/loader.py`): `load_manifest(data: bytes) -> pd.DataFrame`
  — parse the manifest JSON, return its `items` list as a records DataFrame. Image/audio/
  video thus become ordinary DataFrames in `ctx.subject`; **no new context field**.
- **Activity** (`anodyne_workflows/evaluation_activities.py`): for `IMAGE/AUDIO/VIDEO`, load
  via `load_manifest`; run `detect_task(...)`; stamp `ctx.task_type` and
  `ctx.selected_metrics`.
- **Dispatch** (`evaluator.judges_for_modality`): TEXT/TABULAR → existing stat judges **+**
  `TaskMetricsJudge`; GRAPH → GD judges **+** `TaskMetricsJudge`; IMAGE/AUDIO/VIDEO →
  `[QualitativeJudge, TaskMetricsJudge]` **only** (dropping the tabular-distribution judges,
  which assume columnar distributions and don't apply to a manifest — a correctness fix for
  media evaluation). `TaskMetricsJudge`/`QualitativeJudge` still only appear when an LLM is
  configured.
- **Weights** (`models.py`): add `EvalDimension.TASK_QUALITY` to the modality weight groups
  and rebalance (tabular/text and graph groups include it at 0.15 with the others scaled to
  keep the group at 1.0; a `MEDIA_WEIGHTS` group `{TASK_QUALITY: 0.7, QUALITATIVE: 0.3}`).
  The aggregator's renormalization keeps single-dimension runs correct regardless.

### 3.6 API + UI

- **API** (`api-gateway`): `GET /datasets/{dataset_id}/versions/{version_id}/task-metrics`
  → `{ "task_type": str, "available_metrics": [MetricSpec...] }`. It loads the dataset spec,
  runs `detect_task`, and returns `catalog_for`. Tenant-scoped like the other dataset routes.
  The create-evaluation request body gains `selected_metrics: list[str] | None` and
  `task_type: str | None`, passed straight into `EvaluationConfig`.
- **Web** (`apps/web`): the evaluation form fetches the task-metrics endpoint once a version
  is selected, renders a **"Standard metrics"** panel — the detected task-class as a heading
  and a checkbox list of `available_metrics` (all checked by default), using existing brand
  tokens (card/border/muted-foreground, `font-data` for the metric keys). Unchecking excludes
  a metric; the chosen keys post as `selected_metrics`. No new visual language.

### 3.7 Report

`report.py` (JSON already carries `metrics`; HTML): render the `TASK_QUALITY` expert's
`metrics` dict as a labelled **"Standard task metrics"** block, with the task-class named in
the expert's `rationale`. No model/schema change.

## 4. Error handling

- No LLM configured → `TaskMetricsJudge` absent (dimension simply not scored).
- Unknown/`GENERIC` task with no provider, unmet column preconditions, or empty selection →
  `JudgeNotApplicable` (aggregator renormalizes it out).
- LLM output unparseable as the expected JSON → provider raises a domain error caught and
  surfaced as a low/`JudgeNotApplicable` result per provider (never crashes the run); the
  rubric mirrors `QualitativeJudge._parse`.
- Manifest missing/`items` empty → `JudgeNotApplicable`.
- Invalid `task_type` / `selected_metrics` keys in config → 422 at the API boundary
  (validated against `TaskType` / the catalog).

## 5. Testing (TDD)

- **Detection**: `detect_task` per modality + override precedence.
- **Catalog/selection**: catalog is non-empty per task; selection subsets; empty/invalid ⇒
  `JudgeNotApplicable`; score = mean of selected contributions.
- **Each provider**: unit test with a **fake `LLMProvider`** returning canned JSON — assert
  accuracy/macro-F1 on a fixed confusion, faithfulness mapping, intrinsic math (entropy,
  compression, conformance) on fixed frames. Deterministic (`temperature=0`, seeded sample).
- **Loader**: `load_manifest` on image/audio/video manifest bytes → expected columns/rows.
- **Dispatch**: media modalities exclude the stat judges and include `TaskMetricsJudge`;
  tabular/text/graph include it alongside their existing sets; absent LLM ⇒ absent.
- **Aggregator/weights**: `TASK_QUALITY` weighted correctly; single-dimension media run
  renormalizes to that dimension.
- **API**: task-metrics endpoint returns the detected task + catalog; create-evaluation
  round-trips `selected_metrics`/`task_type`; invalid keys → 422.
- **Activity**: one end-to-end per modality with an in-memory store + fake LLM, asserting a
  `TASK_QUALITY` expert appears with the expected metric keys.

## 6. Non-goals / future work

- Multimodal-perception judge (image/audio/video bytes into the LLM; segmentation IoU/Dice;
  ASR audio metrics; CLIP caption adherence) — separate spec; requires an `LLMProvider` port
  extension.
- No new persistence tables: `ExpertScore.metrics` already stores the standard numbers.
- No change to generation, perturbation, or export.

## 7. Global constraints (carried into the plan verbatim)

- `anodyne-core` imports nothing new; LLM access **only** via the `LLMProvider` port.
- Multi-tenant: every DB read stays inside `tenant_session`; the new API route is
  tenant-scoped exactly like sibling dataset routes.
- Deterministic: seeded sampling, `temperature=0`, fixed prompts; a fixed seed reproduces a
  score.
- `mypy --strict` and `ruff` clean; TDD (failing test first); conventional commits ending
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Never log/store plaintext secrets; the LLM model is resolved from the encrypted registry.
