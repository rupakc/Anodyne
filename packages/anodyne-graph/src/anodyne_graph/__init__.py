"""Anodyne Graph: property-graph (knowledge-graph) modality core (walking skeleton).

Canonical model is a typed labelled property graph; the on-disk artifact is
node-link JSON (see `serialization`). GA covers the description -> ontology ->
LLM-generated graph -> JSON artifact happy path; topology/export/eval/UI arrive
in later waves against these frozen interfaces.
"""

from anodyne_graph.constraints import (
    ConstraintReport,
    OntologyConstraintValidator,
    ShaclReport,
    Violation,
    inject_violations,
)
from anodyne_graph.engines import (
    build_graph_engine,
    generate_shard,
    is_from_sample,
    needs_llm,
)
from anodyne_graph.errors import (
    GraphGenerationError,
    OntologyProposalError,
    UnsupportedGraphExportFormatError,
)
from anodyne_graph.export import GRAPH_SUPPORTED_FORMATS, GraphExporter
from anodyne_graph.from_sample import FromSampleGraphGenerator, assert_no_verbatim_subgraph
from anodyne_graph.generator import LLMGraphGenerator
from anodyne_graph.hybrid import HybridGraphGenerator
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
from anodyne_graph.topology import ProceduralTopologyGenerator

__all__ = [
    "GRAPH_SUPPORTED_FORMATS",
    "ConstraintReport",
    "Edge",
    "EdgeType",
    "FromSampleGraphGenerator",
    "GraphDataset",
    "GraphExporter",
    "GraphGenerationError",
    "GraphOntology",
    "HybridGraphGenerator",
    "LLMGraphGenerator",
    "LLMOntologyProposer",
    "Node",
    "NodeType",
    "OntologyConstraintValidator",
    "OntologyProposalError",
    "OntologyProposer",
    "ProceduralTopologyGenerator",
    "PropertySpec",
    "ShaclReport",
    "UnsupportedGraphExportFormatError",
    "Violation",
    "assert_no_verbatim_subgraph",
    "build_graph_engine",
    "compute_metrics",
    "from_json_bytes",
    "generate_shard",
    "inject_violations",
    "is_from_sample",
    "needs_llm",
    "to_json_bytes",
]
