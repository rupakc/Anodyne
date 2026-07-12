from __future__ import annotations

from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import ColumnProfile, Profile, SemanticType
from anodyne_tabular.builder import build_tabular_generator
from anodyne_tabular.copula_generator import CopulaTabularGenerator
from anodyne_tabular.deep_generator import DeepTabularGenerator
from anodyne_tabular.sdv_adapter import SdvNotEnabledError


def _profile() -> Profile:
    return Profile(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        row_count=3,
        columns=[ColumnProfile(name="age", semantic_type=SemanticType.INTEGER, min=0.0, max=9.0)],
        sample_uri="k",
        sample_filename="s.csv",
    )


def _sample() -> pd.DataFrame:
    return pd.DataFrame({"age": [1, 2, 3]})


def test_build_copula_generator_by_default() -> None:
    generator = build_tabular_generator("copula", _profile(), _sample())
    assert isinstance(generator, CopulaTabularGenerator)


@pytest.mark.integration
def test_build_deep_generator_for_ctgan_and_tvae() -> None:
    for kind in ("ctgan", "tvae"):
        generator = build_tabular_generator(kind, _profile(), _sample(), epochs=1)
        assert isinstance(generator, DeepTabularGenerator)


def test_sdv_requires_opt_in() -> None:
    with pytest.raises(SdvNotEnabledError):
        build_tabular_generator("sdv", _profile(), _sample(), enable_sdv=False)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown tabular synthesizer method"):
        build_tabular_generator("nonsense", _profile(), _sample())
