import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.tabular import perturb_tabular


def _table(n: int = 300) -> pa.Table:
    return pa.table(
        {
            "score": pa.array(list(range(n)), type=pa.int64()),
            "name": pa.array([f"n{i}" for i in range(n)]),
        }
    )


def test_nulls_inject_missing_values() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.EDGE_CASE, intensity=0.3, params={"kind": "nulls"}
    )
    out = perturb_tabular(spec, table, seed=1)
    assert out.column("score").null_count > 0
    assert out.column("name").null_count > 0


def test_boundary_values_use_min_max() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.EDGE_CASE,
        intensity=0.5,
        target_fields=["score"],
        params={"kind": "boundary"},
    )
    out = perturb_tabular(spec, table, seed=1)
    vals = [v for v in out.column("score").to_pylist() if v is not None]
    assert 0 in vals and 299 in vals


def test_format_violation_produces_whitespace_strings() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.EDGE_CASE,
        intensity=0.5,
        target_fields=["name"],
        params={"kind": "format"},
    )
    out = perturb_tabular(spec, table, seed=1)
    assert "   " in out.column("name").to_pylist()
