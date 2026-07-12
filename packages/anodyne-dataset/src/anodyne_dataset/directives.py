"""`GenerationDirective` — declarative steering on a `DatasetSpec`.

Requirement 4 (bias/edge-case/use-case directives). Lives alongside `DatasetSpec` (rather than in
`anodyne-generation`) so text/image/audio/video generators (C2-C5) can parse and honor the same
schema without depending on the tabular-specific package. `DatasetSpec.directives` itself stays a
plain `dict[str, object]` (the C0 wire format, already used by storage/gateway/web) -- these
helpers are the only place that knows how to turn it into/from typed `GenerationDirective`s.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class DirectiveKind(StrEnum):
    BIAS = "bias"
    EDGE_CASE = "edge_case"
    USE_CASE = "use_case"


class GenerationDirective(BaseModel):
    """One steering rule applied at generation time.

    - `bias`: skew `field` toward `value` in `rate` fraction of rows (subpopulation bias).
    - `edge_case`: force `field` to an extreme/rare `value` (literal, or symbolic "min"/"max"/
      "null") in `rate` fraction of rows.
    - `use_case`: a named preset (`name`) that behaves like `bias` against `field`/`value`, using
      a built-in default `rate` when omitted (resolved by `anodyne_generation.directives`).
    """

    kind: DirectiveKind
    field: str | None = None
    value: object | None = None
    rate: float | None = None
    name: str | None = None
    params: dict[str, object] = Field(default_factory=dict)


def parse_directives(raw: dict[str, object]) -> list[GenerationDirective]:
    """Parse `DatasetSpec.directives` into typed directives. Missing/empty -> `[]`."""
    items = raw.get("directives", [])
    if not isinstance(items, list):
        raise TypeError("DatasetSpec.directives['directives'] must be a list")
    return [GenerationDirective.model_validate(item) for item in items]


def dump_directives(directives: list[GenerationDirective]) -> dict[str, object]:
    """Serialize typed directives back into the `DatasetSpec.directives` wire format."""
    return {"directives": [d.model_dump(mode="json") for d in directives]}
