from __future__ import annotations

from uuid import uuid4

from anodyne_dataset.directives import DirectiveKind, GenerationDirective, dump_directives
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_generation.directives import DirectiveGenerator
from anodyne_generation.sampler import TabularSampler


def _spec(fields: list[FieldSpec], directives: dict[str, object] | None = None) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=fields,
        target_rows=200,
        directives=directives or {},
    )


def test_no_directives_is_byte_for_byte_passthrough() -> None:
    spec = _spec([FieldSpec(name="flag", semantic_type=SemanticType.BOOLEAN)])
    wrapped = DirectiveGenerator(TabularSampler()).generate(spec, 0, 200, seed=11)
    plain = TabularSampler().generate(spec, 0, 200, seed=11)
    assert wrapped.equals(plain)


def test_deterministic_same_seed_with_directives() -> None:
    directives = [GenerationDirective(kind=DirectiveKind.BIAS, field="flag", value=True, rate=0.9)]
    spec = _spec(
        [FieldSpec(name="flag", semantic_type=SemanticType.BOOLEAN)],
        dump_directives(directives),
    )
    gen = DirectiveGenerator(TabularSampler())
    t1 = gen.generate(spec, 0, 200, seed=3)
    t2 = gen.generate(spec, 0, 200, seed=3)
    assert t1.equals(t2)


def test_disjoint_shards_differ_with_directives() -> None:
    directives = [GenerationDirective(kind=DirectiveKind.BIAS, field="flag", value=True, rate=0.9)]
    spec = _spec(
        [FieldSpec(name="flag", semantic_type=SemanticType.BOOLEAN)],
        dump_directives(directives),
    )
    gen = DirectiveGenerator(TabularSampler())
    a = gen.generate(spec, 0, 200, seed=5).column("flag").to_pylist()
    b = gen.generate(spec, 200, 200, seed=5).column("flag").to_pylist()
    assert a != b
