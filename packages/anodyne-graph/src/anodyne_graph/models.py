"""Domain models for the graph modality (canonical model: typed property graph).

These are the frozen, adapter-free public shapes the rest of the graph waves
(engines, export, evaluation, UI) build on. A graph dataset is a *labelled
property graph* (LPG): typed nodes and typed edges, each carrying an arbitrary
key/value `properties` map. The ontology (T-Box) describes the allowed node and
edge types + their datatype properties; the instance data (A-Box) is the actual
`Node`/`Edge` lists.

Everything here is pure Pydantic + stdlib — no adapter imports, no heavy graph
libraries (networkx/rdflib arrive in later waves as export projections).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PropertySpec(BaseModel):
    """One datatype property on a node/edge type (the graph analog of `FieldSpec`).

    `datatype` is a plain lowercase string — one of
    "string"/"integer"/"float"/"boolean"/"datetime" — kept as a free string
    (not an enum) so later waves can extend the vocabulary without a breaking
    enum change. `constraints` is a free-form dict (e.g. {"choices": [...]},
    {"min": 0}, or {"pii": true} to force faking).
    """

    name: str
    datatype: str = "string"
    nullable: bool = False
    constraints: dict[str, Any] = Field(default_factory=dict)


class NodeType(BaseModel):
    """A node class in the ontology (e.g. "Person", "Company")."""

    name: str
    properties: list[PropertySpec] = Field(default_factory=list)


class EdgeType(BaseModel):
    """A relation type with domain/range (`source_type` -> `target_type`).

    `directed` defaults True; an undirected relation is modelled as a single
    edge with `directed=False` (both waves and exporters read this flag).
    """

    name: str
    source_type: str
    target_type: str
    properties: list[PropertySpec] = Field(default_factory=list)
    directed: bool = True


class GraphOntology(BaseModel):
    """The T-Box: node/edge type schemas + an (unused-in-GA) class hierarchy.

    `subclass_of` maps a node-type name to its parent node-type name; it is a
    forward-compatibility seam for the hierarchy/reasoning waves and is left
    empty by the GA ontology proposer.
    """

    node_types: list[NodeType] = Field(default_factory=list)
    edge_types: list[EdgeType] = Field(default_factory=list)
    subclass_of: dict[str, str] = Field(default_factory=dict)

    def node_type(self, name: str) -> NodeType | None:
        return next((nt for nt in self.node_types if nt.name == name), None)

    def edge_type(self, name: str) -> EdgeType | None:
        return next((et for et in self.edge_types if et.name == name), None)


class Node(BaseModel):
    """A graph node instance: a stable `id`, its ontology `type`, and properties."""

    id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    """A graph edge instance: `source`/`target` are `Node.id` references."""

    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphDataset(BaseModel):
    """A complete generated graph: ontology + instances + summary `metrics`.

    `metrics` carries at least `node_count`, `edge_count`, `nodes_by_type`,
    and `edges_by_type` (see `serialization` / `compute_metrics`).
    """

    ontology: GraphOntology
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def compute_metrics(nodes: list[Node], edges: list[Edge]) -> dict[str, Any]:
    """Summary counts (total + per type) recorded on `GraphDataset.metrics`.

    Kept as a free function so the handler (merging shards) and the generator
    (single shard) compute identical, deterministic metric blobs.
    """
    nodes_by_type: dict[str, int] = {}
    for n in nodes:
        nodes_by_type[n.type] = nodes_by_type.get(n.type, 0) + 1
    edges_by_type: dict[str, int] = {}
    for e in edges:
        edges_by_type[e.type] = edges_by_type.get(e.type, 0) + 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_type": dict(sorted(nodes_by_type.items())),
        "edges_by_type": dict(sorted(edges_by_type.items())),
    }
