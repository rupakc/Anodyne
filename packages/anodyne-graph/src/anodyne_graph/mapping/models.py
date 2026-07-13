"""Domain models for ontology mapping / alignment (SSSOM-shaped).

A `MappingSet` is the adapter-free output of `OntologyAligner`: a list of
`Mapping`s between entities (node types + edge types) of a *source* and a
*target* `GraphOntology`, each carrying a SKOS/SSSOM `predicate`, a confidence
score, a human-readable justification, and the `matcher` that produced it.

These are pure Pydantic + stdlib — no adapter imports, mirroring
`anodyne_graph.models`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MappingRelation(StrEnum):
    """SKOS/SSSOM mapping predicates (the `predicate_id` in SSSOM CURIE form).

    Values are the SSSOM-canonical snake identifiers; `sssom.PREDICATE_CURIE`
    maps each to its `skos:` CURIE for serialization.
    """

    EXACT_MATCH = "exact_match"
    CLOSE_MATCH = "close_match"
    BROAD_MATCH = "broad_match"
    NARROW_MATCH = "narrow_match"
    RELATED_MATCH = "related_match"


class Mapping(BaseModel):
    """One proposed alignment `subject_id --predicate--> object_id`.

    `subject_id`/`object_id` are entity names in the source/target ontology
    respectively; `subject_label`/`object_label` default to those ids.
    `confidence` is in [0, 1]; `matcher` is provenance (e.g. "lexical",
    "lexical+embedding", "llm"). `needs_review` is set by the aligner when the
    confidence lands in the human-review band.
    """

    subject_id: str
    predicate: MappingRelation
    object_id: str
    confidence: float
    justification: str
    matcher: str
    subject_label: str = ""
    object_label: str = ""
    needs_review: bool = False

    @model_validator(mode="after")
    def _default_labels(self) -> Mapping:
        if not self.subject_label:
            self.subject_label = self.subject_id
        if not self.object_label:
            self.object_label = self.object_id
        return self


class MappingSet(BaseModel):
    """A set of `Mapping`s between two ontologies + free-form `metadata`.

    `metadata` records at least the alignment `seed` and thresholds so a run is
    reproducible and self-describing (see `OntologyAligner.align`).
    """

    source_ontology_id: str
    target_ontology_id: str
    mappings: list[Mapping] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
