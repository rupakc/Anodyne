from __future__ import annotations

from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet


def test_mapping_relation_values_match_skos_predicates() -> None:
    assert MappingRelation.EXACT_MATCH.value == "exact_match"
    assert MappingRelation.CLOSE_MATCH.value == "close_match"
    assert MappingRelation.BROAD_MATCH.value == "broad_match"
    assert MappingRelation.NARROW_MATCH.value == "narrow_match"
    assert MappingRelation.RELATED_MATCH.value == "related_match"
    assert {r.value for r in MappingRelation} == {
        "exact_match",
        "close_match",
        "broad_match",
        "narrow_match",
        "related_match",
    }


def test_mapping_defaults() -> None:
    m = Mapping(
        subject_id="Person",
        predicate=MappingRelation.EXACT_MATCH,
        object_id="Human",
        confidence=0.9,
        justification="labels are synonyms",
        matcher="lexical",
    )
    assert m.needs_review is False
    assert m.subject_label == "Person"  # defaults to subject_id when unset
    assert m.object_label == "Human"


def test_mapping_set_holds_ontology_ids_and_mappings() -> None:
    ms = MappingSet(
        source_ontology_id="src",
        target_ontology_id="tgt",
        mappings=[
            Mapping(
                subject_id="Person",
                predicate=MappingRelation.CLOSE_MATCH,
                object_id="Human",
                confidence=0.7,
                justification="j",
                matcher="lexical+embedding",
                needs_review=True,
            )
        ],
        metadata={"seed": 7},
    )
    assert ms.source_ontology_id == "src"
    assert ms.target_ontology_id == "tgt"
    assert len(ms.mappings) == 1
    assert ms.mappings[0].needs_review is True
    assert ms.metadata["seed"] == 7
