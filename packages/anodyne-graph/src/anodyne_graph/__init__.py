"""Anodyne Graph: property-graph (knowledge-graph) modality core (walking skeleton).

Canonical model is a typed labelled property graph; the on-disk artifact is
node-link JSON (see `serialization`). GA covers the description -> ontology ->
LLM-generated graph -> JSON artifact happy path; topology/export/eval/UI arrive
in later waves against these frozen interfaces.
"""

from anodyne_graph.errors import GraphGenerationError, OntologyProposalError
from anodyne_graph.generator import LLMGraphGenerator
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
    compute_metrics,
)
from anodyne_graph.ontology import LLMOntologyProposer
from anodyne_graph.ports import OntologyProposer
from anodyne_graph.serialization import from_json_bytes, to_json_bytes

__all__ = [
    "Edge",
    "EdgeType",
    "GraphDataset",
    "GraphGenerationError",
    "GraphOntology",
    "LLMGraphGenerator",
    "LLMOntologyProposer",
    "Node",
    "NodeType",
    "OntologyProposalError",
    "OntologyProposer",
    "PropertySpec",
    "compute_metrics",
    "from_json_bytes",
    "to_json_bytes",
]
