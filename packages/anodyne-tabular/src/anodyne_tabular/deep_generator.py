"""Higher-fidelity opt-in synthesizer: CTGAN/TVAE (both MIT-licensed, bundled in `ctgan`).

Selectable via `spec.directives["synthesizer"] = "ctgan" | "tvae"`. Heavier than the default
copula generator (GAN/VAE training), so not automatically chosen, but needs no separate license
opt-in (unlike the `sdv` adapter). Same fit-once/sample-per-shard shape as `CopulaTabularGenerator`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import torch
from anodyne_dataset.models import DatasetSpec, FieldSpec, Profile, SemanticType
from anodyne_dataset.ports import Generator
from ctgan import CTGAN, TVAE  # type: ignore[import-untyped]

from anodyne_tabular.constraints import enforce
from anodyne_tabular.copula_generator import _MODELED_TYPES, _fallback_column
from anodyne_tabular.realistic import faker_column
from anodyne_tabular.schema import fields_from_profile

_DISCRETE_TYPES = frozenset({SemanticType.BOOLEAN, SemanticType.CATEGORICAL})


class DeepTabularGenerator(Generator):
    """Fit a CTGAN/TVAE model once (construction), sample per shard (`generate()`)."""

    def __init__(
        self,
        profile: Profile,
        sample: pd.DataFrame,
        *,
        kind: Literal["ctgan", "tvae"] = "ctgan",
        epochs: int = 100,
        fields: list[FieldSpec] | None = None,
    ) -> None:
        if kind not in ("ctgan", "tvae"):
            raise ValueError(f"unknown deep synthesizer kind: {kind!r}; expected 'ctgan' or 'tvae'")
        self._fields = fields or fields_from_profile(profile)
        self._modeled_names = [
            c.name
            for c in profile.columns
            if c.semantic_type in _MODELED_TYPES and c.name in sample.columns
        ]
        discrete = [
            c.name
            for c in profile.columns
            if c.name in self._modeled_names and c.semantic_type in _DISCRETE_TYPES
        ]

        self._model: CTGAN | TVAE | None = None
        if self._modeled_names:
            model_df = sample[self._modeled_names].copy()
            model_cls = CTGAN if kind == "ctgan" else TVAE
            np.random.seed(0)
            torch.manual_seed(0)
            model = model_cls(epochs=epochs)
            model.fit(model_df, discrete)
            self._model = model

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        shard_seed = seed * 1_000_003 + start_row
        arrays: dict[str, pa.Array] = {}

        decoded: pd.DataFrame | None = None
        if self._model is not None and count > 0:
            np.random.seed(shard_seed)
            torch.manual_seed(shard_seed)
            decoded = self._model.sample(count)

        for i, field in enumerate(spec.fields):
            if decoded is not None and field.name in decoded.columns:
                arrays[field.name] = pa.array(decoded[field.name].tolist())
            elif field.semantic_type in _MODELED_TYPES:
                rng = np.random.default_rng([shard_seed, i])
                arrays[field.name] = _fallback_column(field, count, rng)
            else:
                arrays[field.name] = faker_column(field, count, rng_seed=shard_seed + i * 7919)
        table = pa.table(arrays)
        return enforce(table, spec.fields)
