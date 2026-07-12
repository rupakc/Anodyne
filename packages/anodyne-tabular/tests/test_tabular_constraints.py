from __future__ import annotations

import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import FieldSpec, SemanticType
from anodyne_tabular.constraints import enforce


def test_numeric_values_are_clipped_to_bounds() -> None:
    fields = [
        FieldSpec(name="age", semantic_type=SemanticType.INTEGER, constraints={"min": 0, "max": 10})
    ]
    table = pa.table({"age": [-5, 3, 50]})

    result = enforce(table, fields)

    assert result.column("age").to_pylist() == [0, 3, 10]


def test_categorical_out_of_vocab_maps_to_most_frequent_choice() -> None:
    fields = [
        FieldSpec(
            name="plan",
            semantic_type=SemanticType.CATEGORICAL,
            constraints={"choices": ["gold", "silver"], "most_frequent": "gold"},
        )
    ]
    table = pa.table({"plan": ["gold", "bronze", "silver"]})

    result = enforce(table, fields)

    assert result.column("plan").to_pylist() == ["gold", "gold", "silver"]


def test_output_column_order_matches_fields() -> None:
    fields = [
        FieldSpec(name="b", semantic_type=SemanticType.TEXT),
        FieldSpec(name="a", semantic_type=SemanticType.TEXT),
    ]
    table = pa.table({"a": ["x"], "b": ["y"]})

    result = enforce(table, fields)

    assert result.column_names == ["b", "a"]
