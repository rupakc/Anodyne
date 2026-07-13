from __future__ import annotations


class OntologyProposalError(Exception):
    """Raised when ontology proposal cannot parse a usable ontology from the LLM."""


class GraphGenerationError(Exception):
    """Raised when graph generation cannot produce any valid graph for a shard."""


class UnsupportedGraphExportFormatError(ValueError):
    """Raised when `GraphExporter.export`'s `format` isn't a supported graph format."""


class GraphRAGError(ValueError):
    """Raised when a GraphRAG QA fixture cannot be synthesized.

    Typically because the graph is empty/too small to sample the requested
    multi-hop paths, or the sampler cannot find any connected path (a too-sparse
    or fully disconnected graph).
    """
