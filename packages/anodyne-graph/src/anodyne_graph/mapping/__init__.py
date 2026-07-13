"""Ontology mapping / alignment (Track GF).

Given two `GraphOntology` objects (source + target), a hybrid matcher stack
(lexical + embedding + LLM adjudication) proposes entity/relation alignments,
emits **SSSOM** mappings, and routes low-confidence mappings to HITL review.

The domain models (`Mapping`/`MappingSet`) and matcher/aligner logic are
adapter-free; LLM access is only through the `LLMProvider` port and embeddings
through an injected embedding function.
"""

from anodyne_graph.mapping.aligner import (
    AlignmentThresholds,
    OntologyAligner,
    build_mapping_review_task,
    route_to_review,
)
from anodyne_graph.mapping.matchers import (
    EmbeddingMatcher,
    LexicalMatcher,
    LLMJudgement,
    LLMMatcher,
)
from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet
from anodyne_graph.mapping.ports import EmbeddingFn, EntityMatcher
from anodyne_graph.mapping.sssom import (
    PREDICATE_CURIE,
    from_sssom_json,
    from_sssom_tsv,
    to_sssom_json,
    to_sssom_tsv,
)

__all__ = [
    "PREDICATE_CURIE",
    "AlignmentThresholds",
    "EmbeddingFn",
    "EmbeddingMatcher",
    "EntityMatcher",
    "LLMJudgement",
    "LLMMatcher",
    "LexicalMatcher",
    "Mapping",
    "MappingRelation",
    "MappingSet",
    "OntologyAligner",
    "build_mapping_review_task",
    "from_sssom_json",
    "from_sssom_tsv",
    "route_to_review",
    "to_sssom_json",
    "to_sssom_tsv",
]
