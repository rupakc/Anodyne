from __future__ import annotations

from uuid import uuid4

import pytest
from anodyne_dataset.directives import DirectiveKind, GenerationDirective, dump_directives
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_generation.directives import DirectiveError, DirectiveGenerator
from anodyne_generation.sampler import TabularSampler


def _spec(fields: list[FieldSpec], directives: list[GenerationDirective]) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=fields,
        target_rows=500,
        directives=dump_directives(directives),
    )


def _generate(spec: DatasetSpec, count: int, seed: int):  # type: ignore[no-untyped-def]
    return DirectiveGenerator(TabularSampler()).generate(spec, 0, count, seed)


def test_bias_shifts_boolean_distribution_toward_target() -> None:
    field = FieldSpec(name="flag", semantic_type=SemanticType.BOOLEAN)
    baseline = TabularSampler().generate(_spec([field], []), 0, 500, seed=1)
    baseline_rate = sum(baseline.column("flag").to_pylist()) / 500

    directive = GenerationDirective(kind=DirectiveKind.BIAS, field="flag", value=True, rate=0.9)
    biased = _generate(_spec([field], [directive]), 500, seed=1)
    biased_rate = sum(biased.column("flag").to_pylist()) / 500

    assert baseline_rate < 0.65  # sanity: undirected baseline is ~50%
    assert biased_rate >= 0.85


def test_edge_case_forces_numeric_extreme() -> None:
    field = FieldSpec(
        name="age", semantic_type=SemanticType.INTEGER, constraints={"min": 0, "max": 120}
    )
    directive = GenerationDirective(
        kind=DirectiveKind.EDGE_CASE, field="age", value="max", rate=0.2
    )
    n = 2000
    table = _generate(_spec([field], [directive]), n, seed=2)
    at_max = sum(1 for v in table.column("age").to_pylist() if v == 120)

    # Expected ~0.2 * 2000 = 400 (std ~18); 340 is well below that, comfortably
    # ruling out flakiness while still proving the rate took effect.
    assert at_max >= 340


def test_edge_case_null_on_nullable_field() -> None:
    field = FieldSpec(name="note", semantic_type=SemanticType.TEXT, nullable=True)
    directive = GenerationDirective(
        kind=DirectiveKind.EDGE_CASE, field="note", value="null", rate=0.3
    )
    n = 2000
    table = _generate(_spec([field], [directive]), n, seed=4)
    n_null = sum(1 for v in table.column("note").to_pylist() if v is None)

    # Expected ~0.3 * 2000 = 600 (std ~20); 520 leaves ample margin.
    assert n_null >= 520


def test_edge_case_null_on_non_nullable_field_raises() -> None:
    field = FieldSpec(name="note", semantic_type=SemanticType.TEXT, nullable=False)
    directive = GenerationDirective(
        kind=DirectiveKind.EDGE_CASE, field="note", value="null", rate=0.3
    )
    with pytest.raises(DirectiveError):
        _generate(_spec([field], [directive]), 50, seed=4)


def test_use_case_resolves_default_rate_and_applies_like_bias() -> None:
    # "FRAUD" isn't one of the field's normal choices, so its baseline incidence
    # is exactly 0 -- any occurrence comes solely from the directive, letting us
    # measure the resolved default rate (0.02 for "rare_event") cleanly.
    field = FieldSpec(
        name="segment",
        semantic_type=SemanticType.CATEGORICAL,
        constraints={"choices": ["A", "B", "C", "D"]},
    )
    directive = GenerationDirective(
        kind=DirectiveKind.USE_CASE, name="rare_event", field="segment", value="FRAUD"
    )
    table = _generate(_spec([field], [directive]), 2000, seed=6)
    rate = sum(1 for v in table.column("segment").to_pylist() if v == "FRAUD") / 2000

    assert 0.0 < rate <= 0.05  # default "rare_event" rate is 0.02; well below a full bias


def test_unknown_field_raises_directive_error() -> None:
    field = FieldSpec(name="age", semantic_type=SemanticType.INTEGER)
    directive = GenerationDirective(
        kind=DirectiveKind.BIAS, field="does_not_exist", value=1, rate=0.5
    )
    with pytest.raises(DirectiveError):
        _generate(_spec([field], [directive]), 50, seed=7)
