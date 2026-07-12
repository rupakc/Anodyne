from __future__ import annotations

from typing import cast
from uuid import uuid4

from anodyne_dataset.models import ColumnProfile, Profile, SemanticType
from anodyne_tabular.schema import fields_from_profile


def _profile(columns: list[ColumnProfile]) -> Profile:
    return Profile(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        row_count=10,
        columns=columns,
        sample_uri="k",
        sample_filename="s.csv",
    )


def test_numeric_column_gets_min_max_constraints() -> None:
    profile = _profile(
        [ColumnProfile(name="age", semantic_type=SemanticType.INTEGER, min=0.0, max=99.0)]
    )

    fields = fields_from_profile(profile)

    assert fields[0].name == "age"
    assert fields[0].semantic_type is SemanticType.INTEGER
    assert fields[0].constraints == {"min": 0.0, "max": 99.0}


def test_categorical_column_gets_choices_constraint() -> None:
    profile = _profile(
        [
            ColumnProfile(
                name="plan",
                semantic_type=SemanticType.CATEGORICAL,
                categories={"gold": 0.6, "silver": 0.4},
            )
        ]
    )

    fields = fields_from_profile(profile)

    assert set(cast("list[str]", fields[0].constraints["choices"])) == {"gold", "silver"}


def test_nullable_and_ordering_preserved() -> None:
    profile = _profile(
        [
            ColumnProfile(name="a", semantic_type=SemanticType.TEXT, nullable=True),
            ColumnProfile(name="b", semantic_type=SemanticType.EMAIL),
        ]
    )

    fields = fields_from_profile(profile)

    assert [f.name for f in fields] == ["a", "b"]
    assert fields[0].nullable is True
    assert fields[1].nullable is False
