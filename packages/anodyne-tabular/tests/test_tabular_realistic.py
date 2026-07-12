from __future__ import annotations

from anodyne_dataset.models import FieldSpec, SemanticType
from anodyne_tabular.realistic import faker_column


def test_faker_column_is_deterministic() -> None:
    field = FieldSpec(name="name", semantic_type=SemanticType.NAME)

    a = faker_column(field, 20, rng_seed=7)
    b = faker_column(field, 20, rng_seed=7)

    assert a.to_pylist() == b.to_pylist()
    assert len(a) == 20


def test_faker_column_email_looks_like_email() -> None:
    field = FieldSpec(name="email", semantic_type=SemanticType.EMAIL)

    values = faker_column(field, 10, rng_seed=1).to_pylist()

    assert all("@" in v for v in values)


def test_faker_column_different_seeds_differ() -> None:
    field = FieldSpec(name="name", semantic_type=SemanticType.NAME)

    a = faker_column(field, 20, rng_seed=1).to_pylist()
    b = faker_column(field, 20, rng_seed=2).to_pylist()

    assert a != b


def test_mimesis_provider_is_deterministic_and_differs_from_faker() -> None:
    field = FieldSpec(
        name="name", semantic_type=SemanticType.NAME, constraints={"provider": "mimesis"}
    )

    a = faker_column(field, 20, rng_seed=7).to_pylist()
    b = faker_column(field, 20, rng_seed=7).to_pylist()
    assert a == b

    faker_field = FieldSpec(name="name", semantic_type=SemanticType.NAME)
    faker_values = faker_column(faker_field, 20, rng_seed=7).to_pylist()
    assert a != faker_values
