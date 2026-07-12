from __future__ import annotations

import pytest
from anodyne_dataset.directives import (
    DirectiveKind,
    GenerationDirective,
    dump_directives,
    parse_directives,
)
from pydantic import ValidationError


def test_directive_defaults() -> None:
    d = GenerationDirective(kind=DirectiveKind.BIAS, field="age", value=30)
    assert d.rate is None
    assert d.params == {}
    assert d.name is None


def test_one_of_each_kind_constructs() -> None:
    bias = GenerationDirective(kind=DirectiveKind.BIAS, field="plan", value="pro", rate=0.5)
    edge = GenerationDirective(kind=DirectiveKind.EDGE_CASE, field="age", value="max", rate=0.1)
    use_case = GenerationDirective(
        kind=DirectiveKind.USE_CASE, name="rare_event", field="is_fraud", value=True
    )
    assert bias.kind is DirectiveKind.BIAS
    assert edge.kind is DirectiveKind.EDGE_CASE
    assert use_case.kind is DirectiveKind.USE_CASE


def test_parse_directives_empty() -> None:
    assert parse_directives({}) == []
    assert parse_directives({"directives": []}) == []


def test_parse_and_dump_round_trip() -> None:
    directives = [
        GenerationDirective(kind=DirectiveKind.BIAS, field="plan", value="pro", rate=0.3),
        GenerationDirective(kind=DirectiveKind.EDGE_CASE, field="age", value="min", rate=0.05),
    ]
    raw = dump_directives(directives)
    parsed = parse_directives(raw)
    assert parsed == directives


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValidationError):
        GenerationDirective(kind="not-a-kind", field="x")  # type: ignore[arg-type]
