"""Dispatch coverage for `TaskMetricsJudge` across modalities (Task 12).

`judges_for_modality` must append `TaskMetricsJudge` to the TEXT/TABULAR and
GRAPH mixtures whenever an LLM provider + config are available, and media
modalities (IMAGE/AUDIO/VIDEO) must get a dedicated, LLM-only mixture --
`QualitativeJudge` + `TaskMetricsJudge` -- never the tabular-distribution
judges like `FidelityJudge`.
"""

from __future__ import annotations

from anodyne_core.models import LLMResponse, ModelConfig, Usage
from anodyne_dataset.models import Modality
from anodyne_evaluation.evaluator import judges_for_modality
from anodyne_evaluation.graph_judges import StructuralFidelityGraphJudge
from anodyne_evaluation.judges import FidelityJudge, QualitativeJudge
from anodyne_evaluation.judges.task_metrics.judge import TaskMetricsJudge


class _FakeProvider:
    """Injected fake `LLMProvider` -- no network."""

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        return LLMResponse(content="{}", usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def test_text_and_tabular_get_task_metrics_and_keep_fidelity(model_cfg: ModelConfig) -> None:
    provider = _FakeProvider()
    for modality in (Modality.TEXT, Modality.TABULAR):
        judges = judges_for_modality(modality, provider, model_cfg)  # type: ignore[arg-type]
        assert any(isinstance(j, TaskMetricsJudge) for j in judges)
        assert any(isinstance(j, FidelityJudge) for j in judges)


def test_graph_gets_task_metrics_alongside_graph_judges(model_cfg: ModelConfig) -> None:
    provider = _FakeProvider()
    judges = judges_for_modality(Modality.GRAPH, provider, model_cfg)  # type: ignore[arg-type]
    assert any(isinstance(j, StructuralFidelityGraphJudge) for j in judges)
    assert any(isinstance(j, TaskMetricsJudge) for j in judges)


def test_media_modalities_get_exactly_qualitative_and_task_metrics(model_cfg: ModelConfig) -> None:
    provider = _FakeProvider()
    for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO):
        judges = judges_for_modality(modality, provider, model_cfg)  # type: ignore[arg-type]
        types = {type(j) for j in judges}
        assert types == {QualitativeJudge, TaskMetricsJudge}
        assert not any(isinstance(j, FidelityJudge) for j in judges)


def test_no_llm_provider_omits_llm_backed_judges() -> None:
    for modality in (
        Modality.TEXT,
        Modality.TABULAR,
        Modality.GRAPH,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
    ):
        judges = judges_for_modality(modality)
        assert not any(isinstance(j, TaskMetricsJudge) for j in judges)
        assert not any(isinstance(j, QualitativeJudge) for j in judges)


def test_no_model_config_omits_llm_backed_judges() -> None:
    provider = _FakeProvider()
    for modality in (
        Modality.TEXT,
        Modality.TABULAR,
        Modality.GRAPH,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
    ):
        judges = judges_for_modality(modality, provider)  # type: ignore[arg-type]
        assert not any(isinstance(j, TaskMetricsJudge) for j in judges)
        assert not any(isinstance(j, QualitativeJudge) for j in judges)
