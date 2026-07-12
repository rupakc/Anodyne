from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import ColumnProfile, DatasetSpec, Modality, Profile, SemanticType
from anodyne_tabular.sdv_adapter import SdvGaussianCopulaGenerator, SdvNotEnabledError


def _profile() -> Profile:
    return Profile(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        row_count=3,
        columns=[ColumnProfile(name="age", semantic_type=SemanticType.INTEGER, min=0.0, max=99.0)],
        sample_uri="k",
        sample_filename="s.csv",
    )


def test_disabled_by_default_raises_without_importing_sdv() -> None:
    # No `pytest.importorskip` here: this must work even when `sdv` isn't installed,
    # since the whole point of the opt-in gate is refusing *before* the deferred import.
    with pytest.raises(SdvNotEnabledError):
        SdvGaussianCopulaGenerator(_profile(), pd.DataFrame({"age": [1, 2, 3]}), enabled=False)


@pytest.mark.integration
def test_fit_and_sample_smoke() -> None:
    pytest.importorskip("sdv")
    rng = np.random.default_rng(0)
    n = 100
    sample = pd.DataFrame({"age": rng.integers(18, 80, n)})
    profile = _profile()

    generator = SdvGaussianCopulaGenerator(profile, sample, enabled=True)
    spec = DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="sample",
        fields=generator._fields,  # noqa: SLF001
        target_rows=20,
    )

    table = generator.generate(spec, 0, 20, seed=1)

    assert table.num_rows == 20
    assert table.column_names == ["age"]
