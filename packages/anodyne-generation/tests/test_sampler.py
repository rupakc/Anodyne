from uuid import uuid4

from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_generation.sampler import TabularSampler


def _spec(fields: list[FieldSpec], rows: int = 50) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=fields,
        target_rows=rows,
    )


def test_deterministic_same_seed() -> None:
    spec = _spec(
        [
            FieldSpec(
                name="age",
                semantic_type=SemanticType.INTEGER,
                constraints={"min": 0, "max": 120},
            )
        ]
    )
    t1 = TabularSampler().generate(spec, 0, 50, seed=7)
    t2 = TabularSampler().generate(spec, 0, 50, seed=7)
    assert t1.equals(t2)
    assert t1.num_rows == 50 and t1.column_names == ["age"]


def test_integer_constraints_respected() -> None:
    spec = _spec(
        [
            FieldSpec(
                name="age",
                semantic_type=SemanticType.INTEGER,
                constraints={"min": 18, "max": 21},
            )
        ]
    )
    col = TabularSampler().generate(spec, 0, 200, seed=1).column("age").to_pylist()
    assert all(18 <= v <= 21 for v in col)


def test_categorical_uses_choices() -> None:
    spec = _spec(
        [
            FieldSpec(
                name="c",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["a", "b"]},
            )
        ]
    )
    col = set(TabularSampler().generate(spec, 0, 100, seed=2).column("c").to_pylist())
    assert col <= {"a", "b"}


def test_disjoint_ranges_differ() -> None:
    spec = _spec([FieldSpec(name="x", semantic_type=SemanticType.FLOAT)])
    a = TabularSampler().generate(spec, 0, 10, seed=5).column("x").to_pylist()
    b = TabularSampler().generate(spec, 10, 10, seed=5).column("x").to_pylist()
    assert a != b  # different row offset ⇒ different draws
