from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_perturbation.params import (
    BiasParams,
    DriftParams,
    EdgeCaseParams,
    NoiseParams,
    OutlierParams,
    parse_params,
)


def test_defaults_per_family() -> None:
    assert NoiseParams().kind == "gaussian"
    assert DriftParams().kind == "covariate"
    assert OutlierParams().kind == "point"
    assert BiasParams().kind == "class_imbalance"
    assert EdgeCaseParams().kind == "nulls"


def test_parse_params_dispatches_on_family() -> None:
    spec = PerturbationSpec(family=PerturbationFamily.NOISE, params={"kind": "uniform"})
    parsed = parse_params(spec)
    assert isinstance(parsed, NoiseParams)
    assert parsed.kind == "uniform"


def test_parse_params_ignores_unknown_keys() -> None:
    spec = PerturbationSpec(family=PerturbationFamily.BIAS, params={"nonsense": 1})
    parsed = parse_params(spec)
    assert isinstance(parsed, BiasParams)
