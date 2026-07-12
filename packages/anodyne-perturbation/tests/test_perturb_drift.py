import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.tabular import perturb_tabular


def _table(n: int = 400) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "x": pa.array((rng.random(n) * 10).tolist(), type=pa.float64()),
            "label": pa.array((["a", "b", "c"] * n)[:n]),
        }
    )


def test_covariate_drift_shifts_mean() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.DRIFT, params={"kind": "covariate", "shift": 100.0}
    )
    out = perturb_tabular(spec, table, seed=1)
    orig = np.array(table.column("x").to_pylist())
    new = np.array(out.column("x").to_pylist())
    assert abs((new.mean() - orig.mean()) - 100.0) < 1e-6


def test_concept_drift_relabels() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.DRIFT,
        target_fields=["label"],
        params={"kind": "concept", "relabel": {"a": "z"}},
    )
    out = perturb_tabular(spec, table, seed=1)
    new = out.column("label").to_pylist()
    assert "z" in new and "a" not in new


def test_temporal_drift_grows_over_row_order() -> None:
    table = _table()
    spec = PerturbationSpec(
        family=PerturbationFamily.DRIFT, intensity=1.0, params={"kind": "temporal"}
    )
    out = perturb_tabular(spec, table, seed=1)
    orig = np.array(table.column("x").to_pylist())
    new = np.array(out.column("x").to_pylist())
    delta = new - orig
    # Later rows drift more than earlier rows.
    assert delta[-50:].mean() > delta[:50].mean()
