"""Ontology mapping / alignment (Track GF).

Given two `GraphOntology` objects (source + target), a hybrid matcher stack
(lexical + embedding + LLM adjudication) proposes entity/relation alignments,
emits **SSSOM** mappings, and routes low-confidence mappings to HITL review.

The domain models (`Mapping`/`MappingSet`) and matcher/aligner logic are
adapter-free; LLM access is only through the `LLMProvider` port and embeddings
through an injected embedding function.
"""
