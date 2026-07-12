import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation import RegistryPerturbator

_FAMILIES = list(PerturbationFamily)


def _table(n: int = 200) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "num": pa.array((rng.random(n) * 100).tolist(), type=pa.float64()),
            "cat": pa.array((["a", "b", "c"] * n)[:n]),
        }
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_same_seed_identical(family: PerturbationFamily) -> None:
    p = RegistryPerturbator()
    table = _table()
    spec = PerturbationSpec(family=family, intensity=0.5, target_fields=["cat"])
    a = p.perturb(spec, table, "tabular", 42)
    b = p.perturb(spec, table, "tabular", 42)
    assert a.equals(b)


# Drift is a deterministic *distribution* transform (shift/scale/relabel/trend)
# -- its output depends on config, not on the RNG seed -- so it is exempt from
# the "different seed differs" check (it still satisfies same-seed-identical).
_STOCHASTIC = [f for f in _FAMILIES if f is not PerturbationFamily.DRIFT]


@pytest.mark.parametrize("family", _STOCHASTIC)
def test_different_seed_differs(family: PerturbationFamily) -> None:
    p = RegistryPerturbator()
    table = _table()
    spec = PerturbationSpec(family=family, intensity=0.5)
    a = p.perturb(spec, table, "tabular", 1)
    b = p.perturb(spec, table, "tabular", 2)
    assert not a.equals(b)


def test_text_modality_deterministic() -> None:
    p = RegistryPerturbator()
    table = pa.table({"t": pa.array(["hello world foo bar baz"] * 40)})
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.5)
    assert p.perturb(spec, table, "text", 7).equals(p.perturb(spec, table, "text", 7))
    assert not p.perturb(spec, table, "text", 7).equals(p.perturb(spec, table, "text", 8))
