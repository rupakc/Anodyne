"""Default from-sample synthesizer: a Gaussian copula (statistically faithful, fast, permissive).

Modeled columns (numeric/boolean/categorical/datetime) are encoded to a fully numeric frame with
`rdt.HyperTransformer`, a `copulas.multivariate.GaussianMultivariate` is fit on that frame once
(at construction), and each shard's `generate()` reseeds, samples, and reverse-transforms back to
the original types. PII-like columns (name/email/address/text) are never modeled from the sample
-- they're generated fresh via `anodyne_tabular.realistic.faker_column`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, FieldSpec, Profile, SemanticType
from anodyne_dataset.ports import Generator
from copulas.multivariate import GaussianMultivariate  # type: ignore[import-untyped]
from rdt import HyperTransformer  # type: ignore[import-untyped]

from anodyne_tabular.constraints import enforce
from anodyne_tabular.realistic import faker_column
from anodyne_tabular.schema import fields_from_profile

_MODELED_TYPES = frozenset(
    {
        SemanticType.INTEGER,
        SemanticType.FLOAT,
        SemanticType.BOOLEAN,
        SemanticType.CATEGORICAL,
        SemanticType.DATETIME,
    }
)


class CopulaTabularGenerator(Generator):
    """Fit once (construction), sample many times (`generate()`, once per Ray shard)."""

    def __init__(
        self, profile: Profile, sample: pd.DataFrame, fields: list[FieldSpec] | None = None
    ) -> None:
        self._profile = profile
        self._fields = fields or fields_from_profile(profile)
        self._modeled_names = [
            c.name
            for c in profile.columns
            if c.semantic_type in _MODELED_TYPES and c.name in sample.columns
        ]

        self._ht: HyperTransformer | None = None
        self._copula: GaussianMultivariate | None = None
        if self._modeled_names:
            model_df = sample[self._modeled_names].copy()
            ht = HyperTransformer()
            ht.detect_initial_config(data=model_df)
            ht.fit(model_df)
            transformed = ht.transform(model_df)
            copula = GaussianMultivariate()
            copula.fit(transformed)
            self._ht = ht
            self._copula = copula

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        shard_seed = seed * 1_000_003 + start_row
        arrays: dict[str, pa.Array] = {}

        decoded: pd.DataFrame | None = None
        if self._copula is not None and self._ht is not None and count > 0:
            np.random.seed(shard_seed)
            sampled = self._copula.sample(count)
            decoded = self._ht.reverse_transform(sampled)

        for i, field in enumerate(spec.fields):
            if decoded is not None and field.name in decoded.columns:
                arrays[field.name] = pa.array(decoded[field.name].tolist())
            elif field.semantic_type in _MODELED_TYPES:
                # Modeled-type field with no matching modeled column (e.g. added to the
                # schema after the sample was profiled) -- fall back to constraint-driven
                # random sampling rather than nonsensical Faker text for a numeric field.
                rng = np.random.default_rng([shard_seed, i])
                arrays[field.name] = _fallback_column(field, count, rng)
            else:
                arrays[field.name] = faker_column(field, count, rng_seed=shard_seed + i * 7919)
        table = pa.table(arrays)
        return enforce(table, spec.fields)


def _fallback_column(field: FieldSpec, count: int, rng: np.random.Generator) -> pa.Array:
    """Constraint-driven sampling for a modeled-type field absent from the fitted profile."""
    c = field.constraints
    if field.semantic_type is SemanticType.INTEGER:
        lo, hi = int(c.get("min", 0)), int(c.get("max", 100))  # type: ignore[call-overload]
        return pa.array(rng.integers(lo, hi + 1, count).tolist())
    if field.semantic_type is SemanticType.FLOAT:
        lo, hi = float(c.get("min", 0.0)), float(c.get("max", 1.0))  # type: ignore[arg-type]
        return pa.array((rng.random(count) * (hi - lo) + lo).tolist())
    if field.semantic_type is SemanticType.BOOLEAN:
        return pa.array((rng.random(count) < 0.5).tolist())
    choices = list(c.get("choices", ["a", "b", "c"]))  # type: ignore[call-overload]
    idx = rng.integers(0, len(choices), count)
    return pa.array([choices[j] for j in idx])
