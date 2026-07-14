"""End-to-end tests for task-class resolution + media-manifest loading wired
into `run_evaluation` (sub-system F standard metrics).

Mirrors `test_evaluation_activities.py`'s in-memory fakes: no Docker/Ray/LLM
service, just the `sequential_runner` default and a fake `LLMProvider` so the
qualitative + task-quality (standard metrics) judges can run.
"""

from __future__ import annotations

import io
import json
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    Modality,
    SemanticType,
)
from anodyne_evaluation.models import EvaluationReport, EvaluationRun, ExpertScore
from anodyne_workflows.evaluation_activities import (
    EvaluationActivityContext,
    configure_evaluation_activities,
    run_evaluation,
)
from anodyne_workflows.evaluation_workflow import EvaluationInput


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body):  # type: ignore[no-untyped-def]
        self.store[Key] = Body

    def get_object(self, *, Bucket, Key):  # type: ignore[no-untyped-def]
        return {"Body": io.BytesIO(self.store[Key])}


class _FakeDatasetRepo:
    def __init__(
        self, tenant: UUID, dataset: UUID, versions: list[DatasetVersion], modality: Modality
    ) -> None:
        self._versions = versions
        self._spec = DatasetSpec(
            id=dataset,
            tenant_id=tenant,
            name="d",
            description="synthetic dataset",
            modality=modality,
            source="sample",
            fields=[FieldSpec(name="x", semantic_type=SemanticType.FLOAT)],
            target_rows=4,
        )

    async def list_versions(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._versions

    async def get_spec(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._spec


class _FakeEvalRepo:
    def __init__(self) -> None:
        self.runs: dict[UUID, EvaluationRun] = {}
        self.results: dict[UUID, list[ExpertScore]] = {}

    async def create_run(self, run):  # type: ignore[no-untyped-def]
        self.runs[run.id] = run

    async def save_run(self, run):  # type: ignore[no-untyped-def]
        self.runs[run.id] = run

    async def get_run(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return self.runs.get(run_id)

    async def list_runs(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return list(self.runs.values())

    async def add_expert_results(self, tenant_id, run_id, scores):  # type: ignore[no-untyped-def]
        self.results[run_id] = scores

    async def get_expert_results(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return self.results.get(run_id, [])


class _FakeModelRegistry:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    async def get(self, tenant_id, config_id):  # type: ignore[no-untyped-def]
        return self._cfg

    async def list(self, tenant_id):  # type: ignore[no-untyped-def]
        return [self._cfg]


class _FakeLLM:
    """Dispatches on the request's system prompt to satisfy whichever judge is
    asking: the qualitative rubric (`QualitativeJudge`), the text-classification
    label oracle, or the image prompt/label consistency oracle. Every predicted
    label is correct (no wrong-flip) since these tests only assert the
    `task_quality` expert + its metric keys, not exact accuracy values."""

    def __init__(self, text_to_label: dict[str, str] | None = None) -> None:
        self._text_to_label = text_to_label or {}
        self.calls: list[LLMRequest] = []

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        system = request.messages[0].content
        user = request.messages[-1].content
        if "expert data reviewer" in system:
            payload = {"realism": 5, "coherence": 5, "task_fit": 5, "rationale": "ok"}
        elif "labeling short texts" in system:
            lines = [
                line.split(". ", 1)[1]
                for line in user.splitlines()
                if line and line[0].isdigit() and ". " in line
            ]
            labels = [self._text_to_label.get(t, "unknown") for t in lines]
            payload = {"labels": labels}
        elif "image-generation prompt" in system:
            n = user.count("Prompt:")
            payload = {"consistent": [True] * n}
        else:
            raise AssertionError(f"unexpected LLM request, system={system!r}")
        return LLMResponse(content=json.dumps(payload), usage=Usage(total_tokens=1))

    def stream(self, config, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="openai", model="gpt-4o")


def _make_ctx(
    dataset_repo: _FakeDatasetRepo, s3: _FakeS3, llm: _FakeLLM, model_cfg: ModelConfig
) -> EvaluationActivityContext:
    return EvaluationActivityContext(
        repo=_FakeEvalRepo(),  # type: ignore[arg-type]
        dataset_repo=dataset_repo,  # type: ignore[arg-type]
        s3_bucket="anodyne",
        s3_client=s3,
        llm_provider=llm,  # type: ignore[arg-type]
        model_registry=_FakeModelRegistry(model_cfg),
    )


async def test_text_classification_version_resolves_task_quality_expert(
    model_cfg: ModelConfig,
) -> None:
    tenant, dataset = uuid4(), uuid4()
    texts = ["a", "b", "c", "d"]
    labels = ["pos", "neg", "pos", "neg"]
    jsonl = "\n".join(
        json.dumps({"text": t, "label": lbl}) for t, lbl in zip(texts, labels, strict=True)
    ).encode()

    subj = DatasetVersion(
        id=uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/v1.jsonl",
        format="jsonl",
        row_count=len(texts),
    )
    s3 = _FakeS3()
    s3.store[f"{tenant}/{subj.artifact_uri}"] = jsonl

    dataset_repo = _FakeDatasetRepo(tenant, dataset, [subj], Modality.TEXT)
    llm = _FakeLLM(text_to_label=dict(zip(texts, labels, strict=True)))
    ctx = _make_ctx(dataset_repo, s3, llm, model_cfg)
    configure_evaluation_activities(ctx)

    run_id = uuid4()
    inp = EvaluationInput(
        run_id=str(run_id),
        dataset_id=str(dataset),
        tenant_id=str(tenant),
        dataset_version_id=str(subj.id),
        config={"model_config_id": str(model_cfg.id), "sample_rows": 4},
    )
    key = await run_evaluation(inp)

    report = EvaluationReport.model_validate_json(s3.store[f"{tenant}/{key}"])
    task_quality = next(
        (s for s in report.expert_scores if str(s.dimension) == "task_quality"), None
    )
    assert task_quality is not None
    assert {"accuracy", "macro_f1", "class_balance", "duplicate_rate"} <= set(task_quality.metrics)
    assert task_quality.metrics["accuracy"] == pytest.approx(1.0)


async def test_image_manifest_version_resolves_task_quality_expert(
    model_cfg: ModelConfig,
) -> None:
    tenant, dataset = uuid4(), uuid4()
    manifest = {
        "items": [
            {
                "prompt": "cat on mat",
                "label": "cat",
                "object_key": "img/0.png",
                "mime_type": "image/png",
            },
            {
                "prompt": "cat on rug",
                "label": "cat",
                "object_key": "img/1.png",
                "mime_type": "image/png",
            },
            {
                "prompt": "dog in park",
                "label": "dog",
                "object_key": "img/2.png",
                "mime_type": "image/png",
            },
            {
                "prompt": "dog in yard",
                "label": "dog",
                "object_key": "img/3.png",
                "mime_type": "image/png",
            },
        ]
    }
    subj = DatasetVersion(
        id=uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/manifest.json",
        format="json",
        row_count=4,
    )
    s3 = _FakeS3()
    s3.store[f"{tenant}/{subj.artifact_uri}"] = json.dumps(manifest).encode()

    dataset_repo = _FakeDatasetRepo(tenant, dataset, [subj], Modality.IMAGE)
    llm = _FakeLLM()
    ctx = _make_ctx(dataset_repo, s3, llm, model_cfg)
    configure_evaluation_activities(ctx)

    run_id = uuid4()
    inp = EvaluationInput(
        run_id=str(run_id),
        dataset_id=str(dataset),
        tenant_id=str(tenant),
        dataset_version_id=str(subj.id),
        config={"model_config_id": str(model_cfg.id), "sample_rows": 4},
    )
    key = await run_evaluation(inp)

    report = EvaluationReport.model_validate_json(s3.store[f"{tenant}/{key}"])
    task_quality = next(
        (s for s in report.expert_scores if str(s.dimension) == "task_quality"), None
    )
    assert task_quality is not None
    assert {
        "label_balance",
        "prompt_label_consistency",
        "prompt_diversity",
        "duplicate_rate",
    } <= set(task_quality.metrics)
    assert task_quality.metrics["prompt_label_consistency"] == pytest.approx(1.0)
