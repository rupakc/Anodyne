"""Shared, deterministic property synthesis + PII faking for graph engines.

Extracted from the GA ``LLMGraphGenerator`` so every wave-GB engine (topology,
hybrid, from-sample) fakes PII and synthesizes datatype-correct property values
the *same* way. Everything here is pure + seeded: given the same numpy
``Generator`` and Faker state, the output is byte-identical (the platform
determinism contract).

The privacy invariant lives here: a property is PII (its value is *always*
faked, never trusted from an LLM or copied from a sample) when its name matches
a PII keyword or its spec carries ``constraints["pii"] == true``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from faker import Faker

from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.models import GraphOntology, NodeType, PropertySpec

# Substrings that mark a property as PII: its value is always faked, never
# taken from the LLM / copied from a sample (privacy invariant).
PII_KEYWORDS = ("name", "email", "phone", "address", "ssn", "credit_card", "dob", "birth")


def is_pii(prop: PropertySpec) -> bool:
    """True when ``prop`` must be faked (name matches a PII keyword or flagged)."""
    if bool(prop.constraints.get("pii")):
        return True
    lowered = prop.name.lower()
    return any(keyword in lowered for keyword in PII_KEYWORDS)


def coerce(value: Any, datatype: str) -> Any:
    """Best-effort coercion of a provided value to the declared datatype."""
    try:
        if datatype == "integer":
            return int(value)
        if datatype == "float":
            return float(value)
        if datatype == "boolean":
            return bool(value) if not isinstance(value, str) else value.strip().lower() == "true"
    except (TypeError, ValueError):
        pass
    return value


def fake_pii(prop: PropertySpec, fake: Faker) -> Any:
    """Return a faked value for a PII property (never a real/LLM value)."""
    lowered = prop.name.lower()
    if "email" in lowered:
        return fake.email()
    if "phone" in lowered:
        return fake.phone_number()
    if "address" in lowered:
        return fake.address().replace("\n", ", ")
    if "name" in lowered:
        return fake.name()
    if "ssn" in lowered:
        return fake.ssn()
    if "dob" in lowered or "birth" in lowered:
        return fake.date_of_birth().isoformat()
    return fake.word()


def synthesize(prop: PropertySpec, rng: np.random.Generator, fake: Faker) -> Any:
    """Synthesize one datatype-correct value for a non-PII property."""
    choices = prop.constraints.get("choices")
    if isinstance(choices, list) and choices:
        return choices[int(rng.integers(0, len(choices)))]
    if prop.datatype == "integer":
        low = int(prop.constraints.get("min", 0))
        high = int(prop.constraints.get("max", 1000))
        high = max(high, low + 1)
        return int(rng.integers(low, high))
    if prop.datatype == "float":
        return float(rng.random())
    if prop.datatype == "boolean":
        return bool(rng.random() < 0.5)
    if prop.datatype == "datetime":
        return fake.date_time().isoformat()
    return fake.word()


def synthesize_node_properties(
    node_type: NodeType, rng: np.random.Generator, fake: Faker
) -> dict[str, Any]:
    """A full property map for a node with no external values (topology engine).

    PII is faked; everything else is synthesized to its datatype/constraints.
    """
    props: dict[str, Any] = {}
    for prop in node_type.properties:
        props[prop.name] = fake_pii(prop, fake) if is_pii(prop) else synthesize(prop, rng, fake)
    return props


def ontology_from_spec(spec_directives: dict[str, Any]) -> GraphOntology:
    """Load the ontology stored (by GA) in ``spec.directives["ontology"]``.

    Raises:
        GraphGenerationError: if no ontology is present.
    """
    raw = spec_directives.get("ontology")
    if raw is None:
        raise GraphGenerationError(
            "no ontology in directives['ontology']; propose or set one before generating a graph"
        )
    if isinstance(raw, GraphOntology):
        return raw
    return GraphOntology.model_validate(raw)
