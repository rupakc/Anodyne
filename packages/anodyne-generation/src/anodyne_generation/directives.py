"""Applies `GenerationDirective`s (requirement 4) to a generated tabular table.

`DirectiveGenerator` wraps an existing `Generator` (typically `TabularSampler`) and
post-processes its output -- it never rewrites `sampler.py`. A directive-free spec is a
byte-for-byte passthrough of the inner generator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.directives import GenerationDirective, parse_directives
from anodyne_dataset.models import DatasetSpec, FieldSpec
from anodyne_dataset.ports import Generator

#: Default fraction of rows a named `use_case` directive affects when it omits `rate`.
USE_CASE_DEFAULT_RATES: dict[str, float] = {
    "rare_event": 0.02,
    "balanced": 0.5,
    "high_risk_segment": 0.3,
}

_NULL_SENTINEL = "null"
_MIN_SENTINEL = "min"
_MAX_SENTINEL = "max"


class DirectiveError(Exception):
    """Raised when a `GenerationDirective` can't be applied to the given `DatasetSpec`."""


class DirectiveGenerator(Generator):
    """Wraps a `Generator`, applying `spec.directives` to its output table."""

    def __init__(self, inner: Generator) -> None:
        self._inner = inner

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        table = self._inner.generate(spec, start_row, count, seed)
        directives = parse_directives(spec.directives)
        if not directives:
            return table

        fields_by_name = {f.name: f for f in spec.fields}
        for i, directive in enumerate(directives):
            table = self._apply(table, directive, i, fields_by_name, start_row, count, seed)
        return table

    def _apply(
        self,
        table: pa.Table,
        directive: GenerationDirective,
        index: int,
        fields_by_name: dict[str, FieldSpec],
        start_row: int,
        count: int,
        seed: int,
    ) -> pa.Table:
        field_name = directive.field
        if not field_name:
            raise DirectiveError(f"directive {index} ({directive.kind}) is missing a target field")
        field = fields_by_name.get(field_name)
        if field is None:
            raise DirectiveError(
                f"directive {index} targets unknown field {field_name!r}; "
                f"known fields: {sorted(fields_by_name)}"
            )

        rate = self._resolve_rate(directive)
        target = self._resolve_value(directive, field)

        mask_seed = [seed, index, hash(field_name) & 0xFFFFFFFF, start_row]
        mask = np.random.default_rng(mask_seed).random(count) < rate

        column = table.column(field_name).to_pylist()
        for row in range(count):
            if mask[row]:
                column[row] = target
        new_column = pa.array(column, type=table.schema.field(field_name).type)
        col_index = table.schema.get_field_index(field_name)
        return table.set_column(col_index, field_name, new_column)

    def _resolve_rate(self, directive: GenerationDirective) -> float:
        if directive.rate is not None:
            return directive.rate
        if directive.name is not None and directive.name in USE_CASE_DEFAULT_RATES:
            return USE_CASE_DEFAULT_RATES[directive.name]
        raise DirectiveError(
            f"directive has no rate and no resolvable default (name={directive.name!r})"
        )

    def _resolve_value(self, directive: GenerationDirective, field: FieldSpec) -> Any:
        value = directive.value
        if value == _NULL_SENTINEL:
            if not field.nullable:
                raise DirectiveError(
                    f"directive forces field {field.name!r} to null, but it isn't nullable"
                )
            return None
        if value == _MIN_SENTINEL:
            return field.constraints.get("min")
        if value == _MAX_SENTINEL:
            return field.constraints.get("max")
        return value
