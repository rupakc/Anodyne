import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.tabular import perturb_tabular


def _numeric_table(n: int = 500) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "age": pa.array(rng.integers(20, 60, n).tolist(), type=pa.int64()),
            "salary": pa.array((rng.random(n) * 1000).tolist(), type=pa.float64()),
        }
    )


def test_gaussian_noise_shifts_values_but_keeps_mean_within_tolerance() -> None:
    table = _numeric_table()
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.5)
    out = perturb_tabular(spec, table, seed=7)

    orig = np.array(table.column("salary").to_pylist())
    new = np.array(out.column("salary").to_pylist())
    assert not np.allclose(orig, new)  # values actually changed
    # Zero-mean Gaussian noise: population mean barely moves.
    assert abs(orig.mean() - new.mean()) < 0.15 * orig.std()
    # Dispersion grows.
    assert new.std() > orig.std()
    assert out.schema.field("age").type == pa.int64()  # int dtype preserved


def test_categorical_swap_changes_some_values() -> None:
    table = pa.table({"color": pa.array(["red", "blue"] * 100)})
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.5)
    out = perturb_tabular(spec, table, seed=1)
    orig = table.column("color").to_pylist()
    new = out.column("color").to_pylist()
    changed = sum(a != b for a, b in zip(orig, new, strict=True))
    assert 0 < changed < len(orig)
    assert set(new) <= {"red", "blue"}
