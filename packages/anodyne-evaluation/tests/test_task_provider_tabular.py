from __future__ import annotations

import json

# Importing the package registers every provider (tabular included) as a side
# effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for


class _FixedFakeProvider:
    """Fake `LLMProvider` that always returns a fixed `consistent` array, regardless
    of prompt content -- `label_consistency` doesn't have a "true" answer to compare
    against, so the test only cares that the fraction-true reduction is correct."""

    def __init__(self, consistent: list[bool]) -> None:
        self._consistent = consistent
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(
            content=json.dumps({"consistent": self._consistent}), usage=Usage(total_tokens=1)
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


class _ExplodingProvider:
    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM should not be called when label_consistency unselected")

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


# --- Classification -----------------------------------------------------

_CLS_ALL = frozenset({"label_consistency", "class_balance", "feature_completeness"})


def _cls_frame() -> pd.DataFrame:
    # "feature" has one null (index 1) -> feature_completeness = 1 - 1/4 = 0.75.
    # "target" is a balanced 2/2 split -> class_balance (normalized entropy) = 1.0.
    return pd.DataFrame({"feature": ["x1", None, "x3", "x4"], "target": ["A", "A", "B", "B"]})


async def test_tabular_classification_provider_scores_all_metrics(
    model_cfg: ModelConfig,
) -> None:
    llm = _FixedFakeProvider([True, True, False, True])  # 3/4 -> 0.75
    ctx = EvaluationContext(
        subject=_cls_frame(),
        task_type=TaskType.TABULAR_CLASSIFICATION,
        target_field="target",
        sample_rows=4,
    )
    prov = provider_for(TaskType.TABULAR_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_CLS_ALL)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["label_consistency"] == pytest.approx(0.75)
    assert score.metrics["class_balance"] == pytest.approx(1.0)
    assert score.metrics["feature_completeness"] == pytest.approx(0.75)
    expected_score = sum(score.metrics[k] for k in _CLS_ALL) / len(_CLS_ALL)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_tabular_classification_provider_catalog() -> None:
    prov = provider_for(TaskType.TABULAR_CLASSIFICATION)
    assert prov is not None
    catalog = prov.metric_catalog()
    keys = {m.key for m in catalog}
    assert keys == {"label_consistency", "class_balance", "feature_completeness"}
    llm_keys = {m.key for m in catalog if m.requires_llm}
    assert llm_keys == {"label_consistency"}


async def test_tabular_classification_provider_skips_llm_when_not_selected(
    model_cfg: ModelConfig,
) -> None:
    ctx = EvaluationContext(
        subject=_cls_frame(),
        task_type=TaskType.TABULAR_CLASSIFICATION,
        target_field="target",
        sample_rows=4,
    )
    prov = provider_for(TaskType.TABULAR_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"class_balance", "feature_completeness"}),
    )
    assert set(score.metrics) == {"class_balance", "feature_completeness"}
    assert score.score == pytest.approx((1.0 + 0.75) / 2)


async def test_tabular_classification_provider_missing_target_field_raises(
    model_cfg: ModelConfig,
) -> None:
    llm = _FixedFakeProvider([True, True, False, True])
    ctx = EvaluationContext(
        subject=_cls_frame(),
        task_type=TaskType.TABULAR_CLASSIFICATION,
        target_field=None,
        sample_rows=4,
    )
    prov = provider_for(TaskType.TABULAR_CLASSIFICATION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            llm,  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"class_balance"}),
        )


# --- Regression -----------------------------------------------------------

_REG_ALL = frozenset(
    {"target_range_validity", "target_distribution_health", "feature_completeness"}
)


def _reg_frame() -> pd.DataFrame:
    # target has one None (range_validity = 6/7) and a single large outlier that
    # gives it a known, nonzero skew (computed by hand below via pandas .skew()).
    return pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7],
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0, None],
        }
    )


_EXPECTED_SKEW = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 100.0]).skew()
_EXPECTED_HEALTH = 1 - min(1.0, abs(_EXPECTED_SKEW) / 10)


async def test_regression_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(
        subject=_reg_frame(),
        task_type=TaskType.REGRESSION,
        target_field="target",
        sample_rows=7,
    )
    prov = provider_for(TaskType.REGRESSION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]  # regression never calls the LLM
        model_cfg,
        selected=_REG_ALL,
    )
    assert score.dimension.value == "task_quality"
    assert score.metrics["target_range_validity"] == pytest.approx(6 / 7)
    assert score.metrics["target_distribution_health"] == pytest.approx(_EXPECTED_HEALTH)
    assert score.metrics["feature_completeness"] == pytest.approx(1.0)
    expected_score = sum(score.metrics[k] for k in _REG_ALL) / len(_REG_ALL)
    assert score.score == pytest.approx(expected_score)


async def test_regression_provider_catalog_has_no_llm_metrics() -> None:
    prov = provider_for(TaskType.REGRESSION)
    assert prov is not None
    catalog = prov.metric_catalog()
    keys = {m.key for m in catalog}
    assert keys == {
        "target_range_validity",
        "target_distribution_health",
        "feature_completeness",
    }
    assert all(not m.requires_llm for m in catalog)


async def test_regression_provider_range_validity_excludes_infinite(
    model_cfg: ModelConfig,
) -> None:
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [1.0, 2.0, float("inf"), None, 4.0]})
    ctx = EvaluationContext(
        subject=df, task_type=TaskType.REGRESSION, target_field="target", sample_rows=5
    )
    prov = provider_for(TaskType.REGRESSION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"target_range_validity"}),
    )
    assert score.metrics["target_range_validity"] == pytest.approx(3 / 5)


async def test_regression_provider_missing_target_field_raises(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(
        subject=_reg_frame(),
        task_type=TaskType.REGRESSION,
        target_field=None,
        sample_rows=7,
    )
    prov = provider_for(TaskType.REGRESSION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _ExplodingProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"target_range_validity"}),
        )


def test_regression_skew_health_matches_hand_computation() -> None:
    """Pin the exact skew value the health formula is asserted against, so a change
    in the fixture data or pandas' skew implementation surfaces here explicitly."""
    assert _EXPECTED_SKEW == pytest.approx(2.4409670373119305)
    assert _EXPECTED_HEALTH == pytest.approx(0.7559032962688069)
