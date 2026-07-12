import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.tabular import perturb_tabular


def test_point_outliers_create_extreme_values() -> None:
    rng = np.random.default_rng(0)
    table = pa.table({"v": pa.array((rng.normal(0, 1, 500)).tolist(), type=pa.float64())})
    spec = PerturbationSpec(
        family=PerturbationFamily.OUTLIERS, intensity=0.1, params={"magnitude": 6.0}
    )
    out = perturb_tabular(spec, table, seed=3)
    orig = np.array(table.column("v").to_pylist())
    new = np.array(out.column("v").to_pylist())
    # Some cells are pushed well beyond the original range (magnitude=6 sigma).
    assert np.abs(new).max() > np.abs(orig).max() + orig.std()
    n_extreme = int(np.sum(np.abs(new) > np.abs(orig).max() + orig.std()))
    assert n_extreme > 0


def test_contextual_outliers_inject_rare_category() -> None:
    table = pa.table({"cat": pa.array(["a", "b"] * 100)})
    spec = PerturbationSpec(
        family=PerturbationFamily.OUTLIERS,
        intensity=0.2,
        params={"kind": "contextual", "rare_value": "ZZZ"},
    )
    out = perturb_tabular(spec, table, seed=2)
    assert "ZZZ" in out.column("cat").to_pylist()
