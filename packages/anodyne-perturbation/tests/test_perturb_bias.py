import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.tabular import perturb_tabular


def test_class_imbalance_hits_target_ratio() -> None:
    # Balanced 50/50 class field.
    table = pa.table(
        {
            "cls": pa.array(["A", "B"] * 250),
            "x": pa.array(list(range(500)), type=pa.int64()),
        }
    )
    spec = PerturbationSpec(
        family=PerturbationFamily.BIAS,
        target_fields=["cls"],
        params={"target_value": "A", "target_ratio": 0.9},
    )
    out = perturb_tabular(spec, table, seed=5)
    cls = out.column("cls").to_pylist()
    ratio = cls.count("A") / len(cls)
    assert abs(ratio - 0.9) < 0.05
    assert out.num_rows == table.num_rows  # row count preserved


def test_demographic_skew_over_represents_value() -> None:
    table = pa.table({"gender": pa.array(["m", "f"] * 250)})
    spec = PerturbationSpec(
        family=PerturbationFamily.BIAS,
        params={
            "kind": "demographic_skew",
            "field": "gender",
            "target_value": "f",
            "target_ratio": 0.8,
        },
    )
    out = perturb_tabular(spec, table, seed=1)
    g = out.column("gender").to_pylist()
    assert g.count("f") / len(g) > 0.7
