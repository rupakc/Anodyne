from __future__ import annotations


class OntologyProposalError(Exception):
    """Raised when ontology proposal cannot parse a usable ontology from the LLM."""


class GraphGenerationError(Exception):
    """Raised when graph generation cannot produce any valid graph for a shard."""


class UnsupportedGraphExportFormatError(ValueError):
    """Raised when `GraphExporter.export`'s `format` isn't a supported graph format."""
