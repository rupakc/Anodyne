"""SSSOM serialization for `MappingSet` (TSV + JSON), plus inverse parsers.

SSSOM (Simple Standard for Sharing Ontological Mappings) predicates are emitted
as `skos:` CURIEs. The TSV carries the required SSSOM columns plus two
extension columns (`matcher`, `needs_review`) so a round-trip is lossless for
the per-mapping fields; a `#`-commented preamble carries the curie map and the
source/target ontology ids. The JSON form additionally round-trips the full
mapping-set `metadata`.

`to_*` output is deterministic (stable column order, sorted mappings, canonical
JSON). Inverse parsers (`from_*`) exist so alignment artifacts can be reloaded
(e.g. for HITL review or re-export).
"""

from __future__ import annotations

import io
import json
from typing import Any

from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet

_SKOS = "http://www.w3.org/2004/02/skos/core#"

PREDICATE_CURIE: dict[MappingRelation, str] = {
    MappingRelation.EXACT_MATCH: "skos:exactMatch",
    MappingRelation.CLOSE_MATCH: "skos:closeMatch",
    MappingRelation.BROAD_MATCH: "skos:broadMatch",
    MappingRelation.NARROW_MATCH: "skos:narrowMatch",
    MappingRelation.RELATED_MATCH: "skos:relatedMatch",
}
_CURIE_TO_PREDICATE: dict[str, MappingRelation] = {v: k for k, v in PREDICATE_CURIE.items()}

_COLUMNS = [
    "subject_id",
    "subject_label",
    "predicate_id",
    "object_id",
    "object_label",
    "mapping_justification",
    "confidence",
    "matcher",
    "needs_review",
]

_SRC_KEY = "# mapping_set_source_ontology: "
_TGT_KEY = "# mapping_set_target_ontology: "


def _sorted(mappings: list[Mapping]) -> list[Mapping]:
    return sorted(mappings, key=lambda m: (m.subject_id, m.object_id, m.matcher))


def _clean(text: str) -> str:
    """Strip tab/newline so a value stays within one TSV cell."""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _row(m: Mapping) -> dict[str, str]:
    return {
        "subject_id": m.subject_id,
        "subject_label": m.subject_label,
        "predicate_id": PREDICATE_CURIE[m.predicate],
        "object_id": m.object_id,
        "object_label": m.object_label,
        "mapping_justification": _clean(m.justification),
        "confidence": repr(m.confidence),
        "matcher": m.matcher,
        "needs_review": "true" if m.needs_review else "false",
    }


def _mapping_from_row(row: dict[str, str]) -> Mapping:
    return Mapping(
        subject_id=row["subject_id"],
        predicate=_CURIE_TO_PREDICATE.get(row["predicate_id"], MappingRelation.RELATED_MATCH),
        object_id=row["object_id"],
        confidence=float(row["confidence"]),
        justification=row["mapping_justification"],
        matcher=row["matcher"],
        subject_label=row["subject_label"],
        object_label=row["object_label"],
        needs_review=row.get("needs_review", "false").strip().lower() == "true",
    )


def to_sssom_tsv(mapping_set: MappingSet) -> bytes:
    buf = io.StringIO()
    buf.write("# curie_map:\n")
    buf.write(f"#   skos: {_SKOS}\n")
    buf.write(f"{_SRC_KEY}{mapping_set.source_ontology_id}\n")
    buf.write(f"{_TGT_KEY}{mapping_set.target_ontology_id}\n")
    buf.write("\t".join(_COLUMNS) + "\n")
    for m in _sorted(mapping_set.mappings):
        row = _row(m)
        buf.write("\t".join(row[c] for c in _COLUMNS) + "\n")
    return buf.getvalue().encode("utf-8")


def from_sssom_tsv(data: bytes) -> MappingSet:
    source_id = ""
    target_id = ""
    header: list[str] | None = None
    mappings: list[Mapping] = []
    for line in data.decode("utf-8").splitlines():
        if line.startswith(_SRC_KEY):
            source_id = line[len(_SRC_KEY) :]
            continue
        if line.startswith(_TGT_KEY):
            target_id = line[len(_TGT_KEY) :]
            continue
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            continue
        row = dict(zip(header, cells, strict=False))
        mappings.append(_mapping_from_row(row))
    return MappingSet(
        source_ontology_id=source_id,
        target_ontology_id=target_id,
        mappings=mappings,
    )


def _json_mapping(m: Mapping) -> dict[str, Any]:
    return {
        "subject_id": m.subject_id,
        "subject_label": m.subject_label,
        "predicate_id": PREDICATE_CURIE[m.predicate],
        "object_id": m.object_id,
        "object_label": m.object_label,
        "mapping_justification": m.justification,
        "confidence": m.confidence,
        "matcher": m.matcher,
        "needs_review": m.needs_review,
    }


def to_sssom_json(mapping_set: MappingSet) -> bytes:
    payload: dict[str, Any] = {
        "curie_map": {"skos": _SKOS},
        "source_ontology_id": mapping_set.source_ontology_id,
        "target_ontology_id": mapping_set.target_ontology_id,
        "metadata": mapping_set.metadata,
        "mappings": [_json_mapping(m) for m in _sorted(mapping_set.mappings)],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def from_sssom_json(data: bytes) -> MappingSet:
    payload = json.loads(data.decode("utf-8"))
    mappings: list[Mapping] = []
    for raw in payload.get("mappings", []):
        mappings.append(
            Mapping(
                subject_id=raw["subject_id"],
                predicate=_CURIE_TO_PREDICATE.get(
                    raw["predicate_id"], MappingRelation.RELATED_MATCH
                ),
                object_id=raw["object_id"],
                confidence=float(raw["confidence"]),
                justification=raw["mapping_justification"],
                matcher=raw["matcher"],
                subject_label=raw["subject_label"],
                object_label=raw["object_label"],
                needs_review=bool(raw["needs_review"]),
            )
        )
    return MappingSet(
        source_ontology_id=payload.get("source_ontology_id", ""),
        target_ontology_id=payload.get("target_ontology_id", ""),
        mappings=mappings,
        metadata=payload.get("metadata", {}),
    )
