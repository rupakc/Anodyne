from __future__ import annotations

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, FieldSpec, SemanticType
from anodyne_dataset.ports import Generator
from faker import Faker


class TabularSampler(Generator):
    """Deterministic, seeded per-field sampler. Row offset feeds the RNG so shards are disjoint."""

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        columns: dict[str, pa.Array] = {}
        for i, field in enumerate(spec.fields):
            # Independent, reproducible stream per (seed, field, shard offset).
            rng = np.random.default_rng([seed, i, start_row])
            fake = Faker()
            Faker.seed(seed * 1_000_003 + i * 7919 + start_row)
            columns[field.name] = self._column(field, count, rng, fake)
        return pa.table(columns)

    def _column(self, f: FieldSpec, n: int, rng: np.random.Generator, fake: Faker) -> pa.Array:
        c = f.constraints
        st = f.semantic_type
        if st is SemanticType.INTEGER:
            lo, hi = int(c.get("min", 0)), int(c.get("max", 100))  # type: ignore[call-overload]
            return pa.array(rng.integers(lo, hi + 1, n).tolist(), type=pa.int64())
        if st is SemanticType.FLOAT:
            lo, hi = float(c.get("min", 0.0)), float(c.get("max", 1.0))  # type: ignore[arg-type]
            return pa.array((rng.random(n) * (hi - lo) + lo).tolist(), type=pa.float64())
        if st is SemanticType.BOOLEAN:
            return pa.array((rng.random(n) < 0.5).tolist(), type=pa.bool_())
        if st is SemanticType.CATEGORICAL:
            choices = list(c.get("choices", ["a", "b", "c"]))  # type: ignore[call-overload]
            idx = rng.integers(0, len(choices), n)
            return pa.array([choices[j] for j in idx])
        if st is SemanticType.DATETIME:
            return pa.array([fake.date_time().isoformat() for _ in range(n)])
        if st is SemanticType.NAME:
            return pa.array([fake.name() for _ in range(n)])
        if st is SemanticType.EMAIL:
            return pa.array([fake.email() for _ in range(n)])
        if st is SemanticType.ADDRESS:
            return pa.array([fake.address().replace("\n", ", ") for _ in range(n)])
        return pa.array([fake.text(max_nb_chars=80) for _ in range(n)])
