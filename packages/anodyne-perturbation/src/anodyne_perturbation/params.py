"""Typed, per-family parameter models parsed out of `PerturbationSpec.params`.

`PerturbationSpec.params` is an untyped dict (consistent with the rest of the
domain -- `FieldSpec.constraints`, `DatasetSpec.directives`). The family
implementations parse it into one of these models via `parse_params`, so the
corruption logic works against validated, typed config. `extra="ignore"` keeps
a stray key from a UI/API caller from being fatal.
"""

from __future__ import annotations

from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from pydantic import BaseModel, ConfigDict


class _Params(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NoiseParams(_Params):
    # numeric: "gaussian" | "uniform"
    kind: str = "gaussian"
    # string columns in the *text* modality: "char" | "word" | "mask"
    text_op: str = "char"


class DriftParams(_Params):
    # "covariate" (shift/scale numeric) | "concept" (relabel) | "temporal" (trend)
    kind: str = "covariate"
    shift: float | None = None
    scale: float = 1.0
    slope: float = 1.0
    relabel: dict[str, str] = {}


class OutlierParams(_Params):
    # "point" (extreme numeric) | "contextual" (rare category)
    kind: str = "point"
    magnitude: float = 5.0
    rare_value: str = "__ANOMALY__"


class BiasParams(_Params):
    # "class_imbalance" | "demographic_skew"
    kind: str = "class_imbalance"
    field: str | None = None
    target_value: str | None = None
    target_ratio: float = 0.8


class EdgeCaseParams(_Params):
    # "nulls" | "boundary" | "format"
    kind: str = "nulls"


_MODELS: dict[PerturbationFamily, type[_Params]] = {
    PerturbationFamily.NOISE: NoiseParams,
    PerturbationFamily.DRIFT: DriftParams,
    PerturbationFamily.OUTLIERS: OutlierParams,
    PerturbationFamily.BIAS: BiasParams,
    PerturbationFamily.EDGE_CASE: EdgeCaseParams,
}


def parse_params(spec: PerturbationSpec) -> _Params:
    """Parse `spec.params` into the typed param model for `spec.family`."""
    return _MODELS[spec.family].model_validate(spec.params)
