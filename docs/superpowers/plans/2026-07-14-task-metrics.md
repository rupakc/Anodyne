# Per-Modality / Per-Task-Class Standard Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a task-class-aware "standard metrics" layer to the LLM-as-a-Judge evaluator, across all six modalities, with user-selectable metrics per task.

**Architecture:** A new `TASK_QUALITY` mixture-of-experts dimension backed by a `TaskMetricsJudge` that dispatches, by detected `TaskType`, to a per-task `TaskMetricProvider`. Each provider computes intrinsic (non-LLM) statistics plus a sampled, deterministic LLM-oracle pass through the existing `LLMProvider` port, and declares a metric catalog the UI uses for selection. Media modalities become evaluable via a manifest→DataFrame loader. New code is confined to `anodyne-evaluation`, with thin edits to `anodyne-workflows`, `api-gateway`, and `apps/web`.

**Tech Stack:** Python 3.12, Pydantic v2, pandas, numpy, pytest (async), FastAPI, Next.js 16 + Tailwind v4.

## Global Constraints

- `anodyne-core` gets **no** new imports; LLM access **only** via the `anodyne_core.ports.LLMProvider` port (`provider.complete(model_config, LLMRequest)`).
- Every score is normalized `0..1`, higher is better (matches `ExpertScore.score`).
- All LLM calls are deterministic: `params={"temperature": 0}`, fixed prompts, sampling seeded by `ctx.seed`. JSON parsed with the fence-strip regex used by `QualitativeJudge` (` ```json ... ``` `).
- LLM output JSON that cannot be parsed → raise the provider's domain error; the judge converts it to `JudgeNotApplicable` (never crash a run).
- Multi-tenant: the new API route resolves the version through the caller's tenant-scoped repo exactly like sibling routes; no cross-tenant reads.
- `mypy --strict` and `ruff` must pass. TDD: failing test first. Conventional commits ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Test commands run from repo root: `uv run pytest <path>`; types `uv run mypy packages/anodyne-evaluation`; lint `uv run ruff check --fix . && uv run ruff format .`.

---

### Task 1: `TaskType` + `detect_task`

**Files:**
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/task.py`
- Test: `packages/anodyne-evaluation/tests/test_task_detection.py`

**Interfaces:**
- Produces: `TaskType(StrEnum)`; `detect_task(modality: Modality, columns: list[str], *, target_field: str | None = None, target_is_numeric: bool = False) -> TaskType`.

- [ ] **Step 1: Write the failing test**

```python
# test_task_detection.py
from anodyne_dataset.models import Modality
from anodyne_evaluation.task import TaskType, detect_task


def test_text_shapes_map_to_task_types():
    assert detect_task(Modality.TEXT, ["text", "label"]) is TaskType.TEXT_CLASSIFICATION
    assert detect_task(Modality.TEXT, ["question", "answer"]) is TaskType.QA
    assert detect_task(Modality.TEXT, ["document", "summary"]) is TaskType.SUMMARIZATION
    assert detect_task(Modality.TEXT, ["instruction", "response"]) is TaskType.CHAT
    assert detect_task(Modality.TEXT, ["freeform"]) is TaskType.GENERIC


def test_tabular_task_from_target():
    assert detect_task(Modality.TABULAR, ["a", "y"], target_field="y") is TaskType.TABULAR_CLASSIFICATION
    assert detect_task(Modality.TABULAR, ["a", "y"], target_field="y", target_is_numeric=True) is TaskType.REGRESSION
    assert detect_task(Modality.TABULAR, ["a", "b"]) is TaskType.GENERIC


def test_media_tasks_from_label_presence():
    assert detect_task(Modality.IMAGE, ["prompt", "label"]) is TaskType.IMAGE_CLASSIFICATION
    assert detect_task(Modality.IMAGE, ["prompt"]) is TaskType.IMAGE_GENERATION
    assert detect_task(Modality.AUDIO, ["text", "label"]) is TaskType.AUDIO_CLASSIFICATION
    assert detect_task(Modality.AUDIO, ["text"]) is TaskType.SPEECH_SYNTHESIS
    assert detect_task(Modality.VIDEO, ["prompt"]) is TaskType.TEXT_TO_VIDEO
    assert detect_task(Modality.GRAPH, []) is TaskType.GENERIC
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest packages/anodyne-evaluation/tests/test_task_detection.py -q` → ImportError.

- [ ] **Step 3: Implement**

```python
# task.py
from __future__ import annotations

from enum import StrEnum

from anodyne_dataset.models import FieldSpec, Modality, SemanticType
from anodyne_text.shapes import TextShape, detect_shape


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


_TEXT_SHAPE_MAP: dict[TextShape, TaskType] = {
    TextShape.CLASSIFICATION: TaskType.TEXT_CLASSIFICATION,
    TextShape.QA: TaskType.QA,
    TextShape.SUMMARIZATION: TaskType.SUMMARIZATION,
    TextShape.CHAT: TaskType.CHAT,
    TextShape.GENERIC: TaskType.GENERIC,
}


def detect_task(
    modality: Modality,
    columns: list[str],
    *,
    target_field: str | None = None,
    target_is_numeric: bool = False,
) -> TaskType:
    """Infer the task-class from modality + available columns (+ tabular target)."""
    names = set(columns)
    if modality == Modality.TEXT:
        # detect_shape keys on field names only; wrap columns as throwaway FieldSpecs.
        fields = [FieldSpec(name=c, semantic_type=SemanticType.TEXT) for c in columns]
        return _TEXT_SHAPE_MAP[detect_shape(fields)]
    if modality == Modality.TABULAR:
        if target_field is None:
            return TaskType.GENERIC
        return TaskType.REGRESSION if target_is_numeric else TaskType.TABULAR_CLASSIFICATION
    if modality == Modality.IMAGE:
        return TaskType.IMAGE_CLASSIFICATION if "label" in names else TaskType.IMAGE_GENERATION
    if modality == Modality.AUDIO:
        return TaskType.AUDIO_CLASSIFICATION if "label" in names else TaskType.SPEECH_SYNTHESIS
    if modality == Modality.VIDEO:
        return TaskType.TEXT_TO_VIDEO
    return TaskType.GENERIC
```

Add `anodyne-text` to `packages/anodyne-evaluation/pyproject.toml` `dependencies` (workspace member) if not already present.

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): task-class detection (TaskType + detect_task)`.

---

### Task 2: Metric catalog, provider protocol, registry, `TASK_QUALITY` dimension + weights

**Files:**
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/task_metrics.py`
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/models.py` (add dimension + weights)
- Test: `packages/anodyne-evaluation/tests/test_task_metrics_registry.py`

**Interfaces:**
- Consumes: `TaskType` (Task 1), `EvaluationContext` (extended in Task 3 — for this task, only the protocol signature references it via `TYPE_CHECKING`).
- Produces: `MetricSpec`, `TaskMetricProvider` (Protocol), `register_provider(p)`, `provider_for(task) -> TaskMetricProvider | None`, `catalog_for(task) -> list[MetricSpec]`. New enum member `EvalDimension.TASK_QUALITY = "task_quality"`.

- [ ] **Step 1: Write the failing test**

```python
# test_task_metrics_registry.py
from anodyne_evaluation.models import EvalDimension, DEFAULT_WEIGHTS
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, catalog_for, provider_for


def test_task_quality_dimension_has_weight():
    assert EvalDimension.TASK_QUALITY == "task_quality"
    assert DEFAULT_WEIGHTS[EvalDimension.TASK_QUALITY] > 0


def test_unregistered_task_has_no_provider():
    assert provider_for(TaskType.GENERIC) is not None  # generic is registered (Task 4)


def test_metric_spec_shape():
    spec = MetricSpec(key="accuracy", label="Accuracy", description="d", requires_llm=True)
    assert spec.key == "accuracy" and spec.requires_llm is True
```

(The `provider_for(GENERIC) is not None` assertion is satisfied once Task 4 registers the generic provider; keep this test but expect it to pass only after Task 4. For Task 2's own gate, assert `provider_for(TaskType.QA) is None` instead, then update in Task 4.)

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `task_metrics.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider
from pydantic import BaseModel

from anodyne_evaluation.models import ExpertScore
from anodyne_evaluation.task import TaskType

if TYPE_CHECKING:
    from anodyne_evaluation.ports import EvaluationContext


class MetricSpec(BaseModel):
    key: str
    label: str
    description: str
    requires_llm: bool = False


@runtime_checkable
class TaskMetricProvider(Protocol):
    task_type: TaskType

    def metric_catalog(self) -> list[MetricSpec]: ...

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore: ...


_REGISTRY: dict[TaskType, TaskMetricProvider] = {}


def register_provider(p: TaskMetricProvider) -> None:
    _REGISTRY[p.task_type] = p


def provider_for(task: TaskType) -> TaskMetricProvider | None:
    return _REGISTRY.get(task)


def catalog_for(task: TaskType) -> list[MetricSpec]:
    p = _REGISTRY.get(task)
    return p.metric_catalog() if p else []
```

- [ ] **Step 4: Extend `models.py`** — add to `EvalDimension`: `TASK_QUALITY = "task_quality"`. Add a `MEDIA_WEIGHTS` group and fold `TASK_QUALITY` into the tabular/text and graph groups; keep each group summing to ~1.0:

```python
TABULAR_WEIGHTS = {
    EvalDimension.FIDELITY: 0.22, EvalDimension.PRIVACY: 0.17, EvalDimension.UTILITY: 0.17,
    EvalDimension.DIVERSITY: 0.12, EvalDimension.QUALITATIVE: 0.08, EvalDimension.BIAS: 0.09,
    EvalDimension.TASK_QUALITY: 0.15,
}
GRAPH_WEIGHTS = {
    EvalDimension.GRAPH_STRUCTURE: 0.22, EvalDimension.GRAPH_ONTOLOGY: 0.18,
    EvalDimension.GRAPH_PRIVACY: 0.13, EvalDimension.GRAPH_CONNECTIVITY: 0.13,
    EvalDimension.GRAPH_UTILITY: 0.13, EvalDimension.GRAPH_SEMANTIC: 0.08,
    EvalDimension.TASK_QUALITY: 0.13,
}
MEDIA_WEIGHTS = {EvalDimension.TASK_QUALITY: 0.7, EvalDimension.QUALITATIVE: 0.3}
DEFAULT_WEIGHTS = {**TABULAR_WEIGHTS, **GRAPH_WEIGHTS, **MEDIA_WEIGHTS}
```

- [ ] **Step 5: Run tests to verify they pass** (adjust the placeholder assertion per Step 1 note).
- [ ] **Step 6: Commit** — `feat(eval): TASK_QUALITY dimension, metric catalog + provider registry`.

---

### Task 3: `EvaluationContext` / `EvaluationConfig` fields

**Files:**
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/ports.py` (`EvaluationContext`)
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/models.py` (`EvaluationConfig`)
- Test: `packages/anodyne-evaluation/tests/test_evaluation_context_taskfields.py`

**Interfaces:**
- Produces: `EvaluationContext.task_type: TaskType | None = None`, `EvaluationContext.selected_metrics: frozenset[str] | None = None`, `EvaluationContext.graph_qa_items: list[Any] | None = None`. `EvaluationConfig.task_type: str | None`, `EvaluationConfig.selected_metrics: list[str] | None`, `EvaluationConfig.graph_qa_fixture_uri: str | None`.

- [ ] **Step 1: Write the failing test**

```python
from anodyne_evaluation.models import EvaluationConfig
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
import pandas as pd


def test_context_carries_task_fields():
    ctx = EvaluationContext(subject=pd.DataFrame(), task_type=TaskType.QA,
                            selected_metrics=frozenset({"groundedness"}))
    assert ctx.task_type is TaskType.QA
    assert ctx.selected_metrics == frozenset({"groundedness"})


def test_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.task_type is None and cfg.selected_metrics is None
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** In `ports.py` add to the `EvaluationContext` dataclass (import `TaskType`; `Any` for `graph_qa_items` to avoid an import cycle):

```python
    task_type: "TaskType | None" = None
    selected_metrics: frozenset[str] | None = None
    graph_qa_items: list[Any] | None = None
```

In `models.py` `EvaluationConfig` add:

```python
    task_type: str | None = None
    selected_metrics: list[str] | None = None
    graph_qa_fixture_uri: str | None = None
```

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): carry task_type/selected_metrics on eval context+config`.

---

### Task 4: LLM-oracle helpers + `generic` provider

**Files:**
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/judges/task_metrics/__init__.py`
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/judges/task_metrics/base.py`
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/judges/task_metrics/generic.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_generic.py`
- Test: `packages/anodyne-evaluation/tests/conftest.py` (add a `FakeLLMProvider` fixture if not present)

**Interfaces:**
- Consumes: `TaskMetricProvider`, `MetricSpec`, `register_provider` (Task 2); `EvaluationContext` (Task 3); `LLMProvider`, `ModelConfig`, `LLMRequest`, `Message` (`anodyne_core`).
- Produces: `strip_json(raw: str) -> str`; `sample_frame(ctx) -> pd.DataFrame`; `mean_contribution(metrics: dict[str, float], selected: frozenset[str]) -> float`; `TaskMetricError`; `GenericTaskProvider` (registered for `TaskType.GENERIC`).

- [ ] **Step 1: Write the failing test**

```python
import json, pandas as pd, pytest
from anodyne_core.models import LLMResponse, Usage
from anodyne_evaluation.judges.task_metrics.generic import GenericTaskProvider
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for


class FakeLLM:
    def __init__(self, content): self._c = content
    async def complete(self, cfg, req):
        return LLMResponse(content=self._c, usage=Usage())


@pytest.mark.anyio
async def test_generic_provider_scores_rubric(model_cfg):
    llm = FakeLLM(json.dumps({"realism": 4, "coherence": 5, "task_fit": 4, "rationale": "ok"}))
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1, 2]}), task_type=TaskType.GENERIC)
    prov = provider_for(TaskType.GENERIC)
    score = await prov.score(ctx, llm, model_cfg, selected=frozenset({"realism", "coherence", "task_fit"}))
    assert 0.0 <= score.score <= 1.0
    assert set(score.metrics) >= {"realism", "coherence", "task_fit"}
```

(`model_cfg` fixture: a minimal `ModelConfig`; add to `conftest.py`. `anyio` backend already used by existing async tests — mirror them.)

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `base.py`**

```python
from __future__ import annotations

import re
import pandas as pd  # type: ignore[import-untyped]

from anodyne_evaluation.ports import EvaluationContext

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class TaskMetricError(Exception):
    """Raised when a provider's LLM output can't be parsed."""


def strip_json(raw: str) -> str:
    text = raw.strip()
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text


def sample_frame(ctx: EvaluationContext) -> pd.DataFrame:
    n = min(ctx.sample_rows, len(ctx.subject))
    if n <= 0:
        return ctx.subject.head(0)
    return ctx.subject.sample(n=n, random_state=ctx.seed).reset_index(drop=True)


def mean_contribution(metrics: dict[str, float], selected: frozenset[str]) -> float:
    vals = [metrics[k] for k in metrics if k in selected]
    return sum(vals) / len(vals) if vals else 0.0
```

- [ ] **Step 4: Implement `generic.py`** — reuses the qualitative rubric prompt; maps three 1–5 scores to 0..1, keyed `realism/coherence/task_fit`. Register at import. Ensure `judges/task_metrics/__init__.py` imports every provider module so registration is a side effect of importing the package.

- [ ] **Step 5: Run to verify it passes.**
- [ ] **Step 6: Commit** — `feat(eval): task-metrics base helpers + generic provider`.

---

### Task 5: `TaskMetricsJudge`

**Files:**
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/judges/task_metrics/judge.py`
- Test: `packages/anodyne-evaluation/tests/test_task_metrics_judge.py`

**Interfaces:**
- Consumes: `provider_for`, `catalog_for` (Task 2); `EvaluationContext.task_type/selected_metrics` (Task 3); generic provider (Task 4).
- Produces: `TaskMetricsJudge(provider, model_config)` with `dimension = EvalDimension.TASK_QUALITY`.

- [ ] **Step 1: Write the failing test** — assert: (a) with `task_type=GENERIC` and a fake LLM it returns a `TASK_QUALITY` `ExpertScore`; (b) unknown/`None` task with no provider → `JudgeNotApplicable`; (c) `selected_metrics` disjoint from the catalog → `JudgeNotApplicable`; (d) provider `TaskMetricError` → `JudgeNotApplicable`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable
from anodyne_evaluation.task_metrics import catalog_for, provider_for


class TaskMetricsJudge(Judge):
    dimension = EvalDimension.TASK_QUALITY

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        task = ctx.task_type
        if task is None:
            raise JudgeNotApplicable("no task type resolved for this run")
        prov = provider_for(task)
        if prov is None:
            raise JudgeNotApplicable(f"no standard-metric provider for task {task}")
        keys = {m.key for m in catalog_for(task)}
        selected = frozenset(ctx.selected_metrics & keys) if ctx.selected_metrics else frozenset(keys)
        if not selected:
            raise JudgeNotApplicable("no valid standard metrics selected")
        try:
            return await prov.score(ctx, self._provider, self._cfg, selected=selected)
        except TaskMetricError as exc:
            raise JudgeNotApplicable(f"task metrics unavailable: {exc}") from exc
```

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): TaskMetricsJudge dispatching to task providers`.

---

### Task 6: `text_classification` provider

**Files:**
- Create: `packages/anodyne-evaluation/src/anodyne_evaluation/judges/task_metrics/text_classification.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_text_classification.py`

**Interfaces:**
- Consumes: base helpers (Task 4), `MetricSpec`, `register_provider`.
- Produces: `TextClassificationProvider` (`task_type = TaskType.TEXT_CLASSIFICATION`). Catalog keys: `accuracy`, `macro_f1`, `class_balance`, `duplicate_rate`.

- [ ] **Step 1: Write the failing test** — build a 4-row frame `{"text": [...], "label": ["pos","neg","pos","neg"]}`. Fake LLM returns labels matching 3 of 4 (one wrong) as JSON `{"label": "..."}` per call (or one batched call returning a list — pick batched; see impl). Assert `accuracy == 0.75`, `macro_f1` computed from the confusion, `class_balance == 1.0` (perfectly balanced 2/2), `duplicate_rate == 0.0`, and `score == mean_contribution` of selected.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement.** Intrinsic: `class_balance` = normalized Shannon entropy of label counts (`H / log(k)`, 1.0 when uniform, 0 when single class or `k<2`); `duplicate_rate` = `1 - unique(text)/len`. LLM-oracle: one request listing the sampled texts + the candidate label set, asking for a JSON array of predicted labels (`{"labels": ["...", ...]}`), same order; compute `accuracy` and macro-F1 from predicted-vs-stored over the sample. Map each raw metric to a 0..1 contribution (all already 0..1). `score = mean_contribution(metrics, selected)`. Recommendations: flag `accuracy < 0.7` or `class_balance < 0.5`. Register.

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): text-classification standard metrics provider`.

---

### Task 7: `qa`, `summarization`, `chat` providers (text generative family)

**Files:**
- Create: `.../judges/task_metrics/qa.py`, `.../summarization.py`, `.../chat.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_text_generative.py`

**Interfaces:**
- Produces: `QAProvider` (keys `answer_correctness`, `groundedness`, `answerable_rate`, `question_type_diversity`), `SummarizationProvider` (`faithfulness`, `coverage`, `conciseness`, `compression_ratio`, `abstractiveness`), `ChatProvider` (`instruction_following`, `coherence`, `turn_validity`). All register on import.

- [ ] **Step 1: Write the failing test** — one test per provider with a fake LLM returning canned rubric JSON:
  - QA frame `{"question","answer","context"}`; LLM returns `{"correct": [true,true,false], "grounded": [true,...]}` → `answer_correctness == 2/3`; `answerable_rate` = fraction of non-empty answers (intrinsic); precondition: missing `answer` col → `JudgeNotApplicable` (raise `TaskMetricError`? No — precondition failure should surface as `JudgeNotApplicable` from the judge; provider raises `TaskMetricError` for missing columns, which the judge maps). Assert.
  - Summarization frame `{"document","summary"}`; LLM returns 1–5 `{"faithfulness":4,"coverage":5,"conciseness":4}`; assert `compression_ratio` and `abstractiveness` (1 − mean bigram overlap) computed intrinsically; score = mean of selected.
  - Chat frame `{"instruction","response"}`; LLM returns `{"instruction_following":5,"coherence":4}`; `turn_validity` intrinsic (non-empty instruction & response) = 1.0.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement all three.** Each: check required columns (else raise `TaskMetricError`), compute intrinsic + one batched LLM call (`temperature=0`), map 1–5 → `/5`, assemble `metrics`, `score = mean_contribution`. Register each.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): QA/summarization/chat standard metrics providers`.

---

### Task 8: `tabular_classification` + `regression` providers

**Files:**
- Create: `.../judges/task_metrics/tabular.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_tabular.py`

**Interfaces:**
- Produces: `TabularClassificationProvider` (keys `label_consistency`, `class_balance`, `feature_completeness`), `RegressionProvider` (`target_range_validity`, `target_distribution_health`, `feature_completeness`). Both need `ctx.target_field`; missing → `TaskMetricError`.

- [ ] **Step 1: Write the failing test** — classification: frame with a categorical target; fake LLM returns per-row `{"consistent": [true,true,false,true]}` → `label_consistency == 0.75`; `class_balance` = entropy; `feature_completeness` = `1 - mean null rate`. Regression: numeric target; `target_range_validity` = fraction finite & within `[q01, q99]`-derived bounds (define: fraction non-null & finite = 1.0 on clean data), `target_distribution_health` = `1 - clamp(|skew|/10)` (define exact formula in impl and assert on a fixed frame), `feature_completeness`. Assert.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** using `ctx.target_field`. LLM-oracle only for `label_consistency` (sampled rows flattened to `field: value`, ask "is the label consistent with the features?" → JSON bool array). Register.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): tabular classification/regression standard metrics providers`.

---

### Task 9: media providers (`image`, `audio`, `video`)

**Files:**
- Create: `.../judges/task_metrics/media.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_media.py`

**Interfaces:**
- Produces: `ImageTaskProvider` (registered for `IMAGE_CLASSIFICATION` and `IMAGE_GENERATION`; keys `label_balance`, `prompt_label_consistency`, `prompt_diversity`, `duplicate_rate`), `AudioTaskProvider` (`AUDIO_CLASSIFICATION`/`SPEECH_SYNTHESIS`; keys `label_balance`, `transcript_label_consistency`, `duration_uniformity`, `transcript_quality`), `VideoTaskProvider` (`TEXT_TO_VIDEO`; keys `duration_conformance`, `resolution_consistency`, `fps_consistency`, `prompt_diversity`, `prompt_quality`).

Provider registration for multiple task types: `register_provider` keys on `p.task_type`; register the same instance under both keys by setting `task_type` per registration — simplest is two thin subclasses (e.g. `ImageClassificationProvider`/`ImageGenerationProvider`) sharing an implementation base, each with its own `task_type` and a `metric_catalog` filtered to the keys valid for that task (classification includes label keys; generation omits them).

- [ ] **Step 1: Write the failing test** — image classification manifest frame `{"prompt","label","object_key","mime_type"}`; fake LLM returns `{"consistent":[...]}` → `prompt_label_consistency`; `label_balance` entropy; `prompt_diversity` = `unique(prompt)/len`; `duplicate_rate`. Video frame `{"prompt","duration_seconds","width","height","fps"}`; `resolution_consistency`/`fps_consistency` = fraction equal to the modal value; `duration_conformance` = fraction within ±10% of the median. Audio frame `{"text","label","duration_seconds"}`. Assert numeric values on fixed frames.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Intrinsic conformance helpers operate on the manifest DataFrame columns. `prompt_quality`/`transcript_quality` = one 1–5 LLM rubric over sampled prompts/transcripts (`/5`). Guard missing columns with `TaskMetricError`. Register all task-type variants.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): image/audio/video standard metrics providers`.

---

### Task 10: `graph_qa` provider

**Files:**
- Create: `.../judges/task_metrics/graph_qa.py`
- Test: `packages/anodyne-evaluation/tests/test_task_provider_graph_qa.py`

**Interfaces:**
- Consumes: `ctx.subject_graph` (`GraphDataset`), `ctx.graph_qa_items` (`list[GraphQAItem]`, Task 3). `anodyne_graph.graphrag.models.GraphQAItem`, `QAPath`.
- Produces: `GraphQAProvider` (`task_type = TaskType.GRAPH_QA`; keys `answer_groundedness`, `multi_hop_correctness`, `answerable_rate`, `question_clarity`). Absent `graph_qa_items` → `TaskMetricError` (→ `JudgeNotApplicable`).

- [ ] **Step 1: Write the failing test** — a tiny `GraphDataset` (2–3 nodes, 1–2 edges) + two `GraphQAItem`s (one whose `gold_path` edges/nodes all exist and resolve, one whose terminal node id is missing). Assert: `answerable_rate` = fraction of items whose `gold_path` fully resolves on the graph (0.5); `multi_hop_correctness` = fraction where re-traversing `gold_path` from `start_node_id` reaches `terminal_node_id` (intrinsic, graph-derived, no LLM); `answer_groundedness` = fraction whose `answer_node_ids` all exist as node ids. `question_clarity` = 1–5 LLM rubric over the question surface forms (fake LLM → fixed). Score = mean of selected.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Build node-id and adjacency indices from `ctx.subject_graph`; the three graph metrics are pure graph checks; only `question_clarity` uses the LLM. Register.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): graph multi-hop QA standard metrics provider`.

---

### Task 11: `load_manifest` loader

**Files:**
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/loader.py`
- Test: `packages/anodyne-evaluation/tests/test_loader_manifest.py`

**Interfaces:**
- Produces: `load_manifest(data: bytes) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test** — image/audio/video manifest bytes (`json.dumps({"items": [ ... ]}).encode()` and also the bare-list form `[ ... ]`) → DataFrame with the expected columns and row count; empty `items` → empty DataFrame (no error).

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement**

```python
def load_manifest(data: bytes) -> pd.DataFrame:
    """Parse a media dataset's manifest JSON into a records DataFrame.

    Accepts either ``{"items": [...]}`` or a bare ``[...]`` list.
    """
    doc = json.loads(data.decode("utf-8"))
    items = doc["items"] if isinstance(doc, dict) else doc
    return pd.DataFrame.from_records(items or [])
```

(Add `import json` at the top of `loader.py`.)

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): load media manifests into a DataFrame for evaluation`.

---

### Task 12: dispatch — `judges_for_modality` + media branch

**Files:**
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/evaluator.py`
- Test: `packages/anodyne-evaluation/tests/test_judge_dispatch_task_metrics.py`

**Interfaces:**
- Consumes: `TaskMetricsJudge` (Task 5), `QualitativeJudge`.
- Produces: updated `judges_for_modality(modality, provider, model_config)` and a new `media_judges(provider, model_config)`.

- [ ] **Step 1: Write the failing test** — with a provider+cfg present: TEXT and TABULAR judge lists contain a `TaskMetricsJudge` **and** the existing `FidelityJudge` etc.; GRAPH contains graph judges **and** `TaskMetricsJudge`; IMAGE/AUDIO/VIDEO contain exactly `{QualitativeJudge, TaskMetricsJudge}` and **no** `FidelityJudge`. With `provider=None`: none of the lists contain `TaskMetricsJudge` or `QualitativeJudge`.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Append `TaskMetricsJudge(provider, model_config)` to `default_judges` and `graph_judges` outputs only when both are non-None (mirror the existing qualitative/semantic guard). Add:

```python
def media_judges(provider=None, model_config=None) -> list[Judge]:
    judges: list[Judge] = []
    if provider is not None and model_config is not None:
        judges.append(QualitativeJudge(provider, model_config))
        judges.append(TaskMetricsJudge(provider, model_config))
    return judges
```

In `judges_for_modality`: `GRAPH` → graph set; `IMAGE/AUDIO/VIDEO` → `media_judges`; else default set. Import `TaskMetricsJudge` from `judges/task_metrics/judge.py` and ensure `import anodyne_evaluation.judges.task_metrics` runs so providers register.

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): dispatch TaskMetricsJudge per modality; media judge set`.

---

### Task 13: activity wiring

**Files:**
- Modify: `packages/anodyne-workflows/src/anodyne_workflows/evaluation_activities.py`
- Test: `packages/anodyne-workflows/tests/test_evaluation_activity_task_metrics.py` (mirror the existing evaluation activity test style)

**Interfaces:**
- Consumes: `load_manifest`, `detect_task`, `load_graphrag_qa` (define inline: parse `graph_qa_fixture_uri` bytes into `list[GraphQAItem]`).

- [ ] **Step 1: Write the failing test** — end-to-end `run_evaluation` for (a) a TEXT classification version (JSONL) and (b) an IMAGE version (manifest) using an in-memory object store, a fake `DatasetRepository` returning a spec with the right modality, `ctx.runner = sequential_runner`, and a fake LLM. Assert the persisted report contains a `TASK_QUALITY` expert with the expected metric keys.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** After computing `modality`: for `IMAGE/AUDIO/VIDEO` set `subject = load_manifest(bytes)`. Compute `columns = list(subject.columns)`, `target_is_numeric = target_field in numeric dtypes`, then:

```python
task_type = TaskType(cfg.task_type) if cfg.task_type else detect_task(
    modality, columns, target_field=cfg.target_field, target_is_numeric=target_is_numeric)
graph_qa_items = None
if cfg.graph_qa_fixture_uri:
    graph_qa_items = load_graphrag_qa(await store.get(cfg.graph_qa_fixture_uri))
```

Pass `task_type`, `selected_metrics=frozenset(cfg.selected_metrics) if cfg.selected_metrics else None`, and `graph_qa_items` into `EvaluationContext(...)`.

- [ ] **Step 4: Run to verify it passes** — `uv run pytest packages/anodyne-workflows/tests/test_evaluation_activity_task_metrics.py -q`.
- [ ] **Step 5: Commit** — `feat(workflows): resolve task-class + load manifests in run_evaluation`.

---

### Task 14: report HTML — standard-metrics block

**Files:**
- Modify: `packages/anodyne-evaluation/src/anodyne_evaluation/report.py`
- Test: `packages/anodyne-evaluation/tests/test_evaluation_report.py` (add a case)

**Interfaces:**
- Consumes: an `EvaluationReport` whose `expert_scores` include a `TASK_QUALITY` entry with a populated `metrics` dict.

- [ ] **Step 1: Write the failing test** — render an HTML report for a report containing a `TASK_QUALITY` `ExpertScore` with `metrics={"accuracy":0.9,"macro_f1":0.88}`; assert the HTML contains a "Standard task metrics" heading and both metric keys/values.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** — when a `TASK_QUALITY` expert is present, emit a labelled block listing its `metrics` (key → formatted value). Keep the existing JSON renderer unchanged (already carries `metrics`).
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(eval): render standard task metrics in the HTML report`.

---

### Task 15: API — task-metrics catalog route + config passthrough

**Files:**
- Modify: `apps/api-gateway/src/api_gateway/evaluation_routes.py`
- Test: `apps/api-gateway/tests/test_evaluation_routes_task_metrics.py` (mirror existing route tests)

**Interfaces:**
- Consumes: `detect_task`, `catalog_for`, `DatasetRepository.get_spec`/`get_version`.
- Produces: `GET /datasets/{dataset_id}/versions/{version_id}/task-metrics` → `{"task_type": str, "available_metrics": [MetricSpec...]}`; `EvaluateRequest.selected_metrics: list[str] | None`, `EvaluateRequest.task_type: str | None`, threaded into `_build_config`.

- [ ] **Step 1: Write the failing test** — with a fake tenant-scoped `DatasetRepository` returning a TEXT spec whose fields are `text,label`, `GET .../task-metrics` returns `task_type == "text_classification"` and a non-empty `available_metrics`; POST `.../evaluate` with `selected_metrics=["accuracy"]` round-trips into the workflow input config; unknown version → 404.

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** The route loads the spec (tenant-scoped), derives `columns` from `spec.fields`, runs `detect_task`, returns `catalog_for`. Add the two request fields; in `_build_config` add them when non-None. Import `import anodyne_evaluation.judges.task_metrics` so the catalog registry is populated in the gateway process.

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(api): task-metrics catalog endpoint + selected_metrics passthrough`.

---

### Task 16: Web UI — "Standard metrics" selection panel

**Files:**
- Modify: the evaluation-launch form component under `apps/web/app/**` (locate the component that POSTs to `/evaluate`; likely `apps/web/app/(app)/**/evaluate` or an evaluation dialog). Confirm exact path with `rg -l "evaluate" apps/web/app`.
- Modify: the web API client that wraps evaluation calls (mirror the existing `startEvaluation` call).
- Test: component/interaction test if the web package has a test setup (`apps/web/**/*.test.tsx`); otherwise a manual verification note + `npm run build` type-check.

**Interfaces:**
- Consumes: `GET .../task-metrics`; POST `.../evaluate` body now accepts `selected_metrics`.

- [ ] **Step 1:** Confirm the evaluation form path and existing fetch client (`rg -n "evaluate|task-metrics|selected_metrics" apps/web`).
- [ ] **Step 2: Write/extend a test** if a harness exists (assert the panel renders fetched metrics as checked checkboxes and that unchecking removes the key from the submitted body). If no web test harness, skip to Step 3 and rely on `npm run build`/typecheck.
- [ ] **Step 3: Implement.** When a version is selected, fetch the task-metrics endpoint; render a **"Standard metrics"** panel: the `task_type` as a heading (humanized) and a checkbox list of `available_metrics` (all checked by default), styled with existing tokens (`border-border`, `bg-card`, `text-muted-foreground`, `font-[family-name:var(--font-data)]` for the metric `key`). Submit the checked keys as `selected_metrics`. Graceful empty state when `available_metrics` is empty (hide the panel).
- [ ] **Step 4: Verify** — `cd apps/web && npm run build` (or the repo's web typecheck/lint) passes; if a test harness exists, `npm test` passes.
- [ ] **Step 5: Commit** — `feat(web): per-task standard-metrics selection in the evaluation form`.

---

## Final verification

- [ ] `uv run ruff check --fix . && uv run ruff format .`
- [ ] `uv run mypy packages/anodyne-evaluation packages/anodyne-workflows apps/api-gateway`
- [ ] `uv run pytest packages/anodyne-evaluation packages/anodyne-workflows apps/api-gateway -q`
- [ ] Whole-branch code review, then `superpowers:finishing-a-development-branch`.
