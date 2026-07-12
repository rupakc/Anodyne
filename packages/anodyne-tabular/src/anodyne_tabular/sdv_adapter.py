"""Opt-in SDV adapter (`sdv` is BSL 1.1 -- separately licensed, not the permissive default).

`sdv` is never imported at module load time (so `anodyne_tabular` loads fine without it
installed) and this generator refuses to run unless explicitly enabled by the operator
(`enabled=True`, wired from the `ANODYNE_TABULAR_ENABLE_SDV` worker setting) *and* the tenant
requests it (`directives["synthesizer"] = "sdv"`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, FieldSpec, Profile
from anodyne_dataset.ports import Generator

from anodyne_tabular.constraints import enforce
from anodyne_tabular.copula_generator import _MODELED_TYPES, _fallback_column
from anodyne_tabular.realistic import faker_column
from anodyne_tabular.schema import fields_from_profile


class SdvNotEnabledError(RuntimeError):
    """Raised when the SDV adapter is requested without being explicitly opted into."""


class SdvGaussianCopulaGenerator(Generator):
    """Fit an `sdv.single_table.GaussianCopulaSynthesizer` once, sample per shard."""

    def __init__(
        self,
        profile: Profile,
        sample: pd.DataFrame,
        fields: list[FieldSpec] | None = None,
        *,
        enabled: bool,
    ) -> None:
        if not enabled:
            raise SdvNotEnabledError(
                "SDV is a separately-licensed (BSL 1.1) opt-in adapter; install "
                "anodyne-tabular[sdv] and set ANODYNE_TABULAR_ENABLE_SDV=true to use it"
            )
        # Deferred import: keeps `sdv` (and its BSL 1.1 license) entirely out of the
        # default install/import path.
        from sdv.metadata import SingleTableMetadata  # type: ignore[import-not-found]
        from sdv.single_table import GaussianCopulaSynthesizer  # type: ignore[import-not-found]

        self._fields = fields or fields_from_profile(profile)
        self._modeled_names = [
            c.name
            for c in profile.columns
            if c.semantic_type in _MODELED_TYPES and c.name in sample.columns
        ]

        self._synth: Any | None = None
        if self._modeled_names:
            model_df = sample[self._modeled_names].copy()
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=model_df)
            synth = GaussianCopulaSynthesizer(metadata)
            synth.fit(data=model_df)
            self._synth = synth

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        shard_seed = seed * 1_000_003 + start_row
        arrays: dict[str, pa.Array] = {}

        decoded: pd.DataFrame | None = None
        if self._synth is not None and count > 0:
            np.random.seed(shard_seed)
            decoded = self._synth.sample(num_rows=count)

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
