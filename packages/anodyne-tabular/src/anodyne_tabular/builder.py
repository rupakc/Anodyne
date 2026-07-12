"""Dispatch: pick and fit a `Generator` for a from-sample dataset by synthesizer method name."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
from anodyne_dataset.models import Profile
from anodyne_dataset.ports import Generator

from anodyne_tabular.copula_generator import CopulaTabularGenerator
from anodyne_tabular.deep_generator import DeepTabularGenerator
from anodyne_tabular.sdv_adapter import SdvGaussianCopulaGenerator

_METHODS = ("copula", "ctgan", "tvae", "sdv")


def build_tabular_generator(
    method: str,
    profile: Profile,
    sample: pd.DataFrame,
    *,
    epochs: int = 100,
    enable_sdv: bool = False,
) -> Generator:
    """Fit and return the `Generator` for `method` ("copula" default / "ctgan" / "tvae" / "sdv")."""
    if method == "copula":
        return CopulaTabularGenerator(profile, sample)
    if method in ("ctgan", "tvae"):
        return DeepTabularGenerator(profile, sample, kind=method, epochs=epochs)  # type: ignore[arg-type]
    if method == "sdv":
        return SdvGaussianCopulaGenerator(profile, sample, enabled=enable_sdv)
    raise ValueError(f"unknown tabular synthesizer method: {method!r}; expected one of {_METHODS}")
