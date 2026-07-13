from __future__ import annotations

import json

from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet
from anodyne_graph.mapping.sssom import (
    from_sssom_json,
    from_sssom_tsv,
    to_sssom_json,
    to_sssom_tsv,
)

_REQUIRED_COLUMNS = [
    "subject_id",
    "predicate_id",
    "object_id",
    "mapping_justification",
    "confidence",
    "subject_label",
    "object_label",
]


def _mapping_set(metadata: dict[str, object] | None = None) -> MappingSet:
    return MappingSet(
        source_ontology_id="src",
        target_ontology_id="tgt",
        mappings=[
            Mapping(
                subject_id="Person",
                predicate=MappingRelation.EXACT_MATCH,
                object_id="Person",
                confidence=1.0,
                justification="identical labels",
                matcher="lexical",
            ),
            Mapping(
                subject_id="Organization",
                predicate=MappingRelation.CLOSE_MATCH,
                object_id="Organisation",
                confidence=1.0 / 3.0,
                justification="near synonym",
                matcher="lexical+embedding",
                needs_review=True,
            ),
        ],
        metadata=metadata or {},
    )


def test_tsv_has_required_columns_and_skos_curies() -> None:
    data = to_sssom_tsv(_mapping_set())
    text = data.decode("utf-8")
    body = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header = body[0].split("\t")
    for col in _REQUIRED_COLUMNS:
        assert col in header
    # predicate_id column uses skos CURIEs.
    assert "skos:exactMatch" in text
    assert "skos:closeMatch" in text


def _by_subject(ms: MappingSet) -> dict[str, dict[str, object]]:
    return {m.subject_id: m.model_dump() for m in ms.mappings}


def test_tsv_round_trip_preserves_mappings() -> None:
    ms = _mapping_set()
    restored = from_sssom_tsv(to_sssom_tsv(ms))
    assert restored.source_ontology_id == "src"
    assert restored.target_ontology_id == "tgt"
    # Serialization canonicalizes ordering; compare order-independently.
    assert _by_subject(restored) == _by_subject(ms)


def test_json_round_trip_is_lossless() -> None:
    ms = _mapping_set(metadata={"seed": 3, "auto_accept": 0.85})
    restored = from_sssom_json(to_sssom_json(ms))
    assert restored.metadata == ms.metadata
    assert restored.source_ontology_id == ms.source_ontology_id
    assert restored.target_ontology_id == ms.target_ontology_id
    assert _by_subject(restored) == _by_subject(ms)


def test_json_is_valid_and_carries_curie_map() -> None:
    payload = json.loads(to_sssom_json(_mapping_set()).decode("utf-8"))
    assert payload["curie_map"]["skos"].startswith("http")
    assert len(payload["mappings"]) == 2
    assert payload["mappings"][0]["predicate_id"].startswith("skos:")


def test_deterministic_bytes_across_two_serializations() -> None:
    ms = _mapping_set(metadata={"seed": 1})
    assert to_sssom_tsv(ms) == to_sssom_tsv(ms)
    assert to_sssom_json(ms) == to_sssom_json(ms)
