"""`GraphExporter`: reads a `graph_json` artifact and serializes it to the
RDF/semantic, property-graph, and GNN interchange formats sub-system GC adds
(extends sub-system E's `Exporter` port — see `anodyne_export.exporter`).

Every format is produced from the single in-memory `GraphDataset` parsed via
`anodyne_graph.serialization.from_json_bytes` -- the artifact is loaded once
and never materialized twice (GC's chunking concession: for GA/GB graph
sizes the whole graph fits in memory, so "stream" here means "write each
format's encoder its own single pass over `nodes`/`edges`", not a Parquet-style
batch iterator).

**RDF mapping.** Nodes become subjects typed `rdf:type` onto their ontology
class (`onto:<NodeType>`); node/edge properties become datatype-property
triples under the `onto:` namespace; an edge becomes a single
`(subject onto:<EdgeType> object)` triple. Edge *properties* have no direct
RDF encoding (a triple has no attribute slots), so GC uses **standard RDF
reification** (`rdf:Statement`/`rdf:subject`/`rdf:predicate`/`rdf:object`),
not RDF-star -- reification round-trips through every serializer GC supports
(Turtle/N-Triples/JSON-LD/RDF-XML) without a star-aware parser, at the cost of
extra triples. The reification blank node is deterministically named
`stmt-<edge.id>` (not a random `BNode`) so output is stable across runs.

**OWL mapping.** Node types -> `owl:Class` (+ `rdfs:subClassOf` from
`GraphOntology.subclass_of`); edge types -> `owl:ObjectProperty` with
`rdfs:domain`/`rdfs:range` from `source_type`/`target_type`; each
node/edge-type `PropertySpec` -> an `owl:DatatypeProperty` named
`onto:<Type>.<property>` with `rdfs:domain`/`rdfs:range` (range from
`PropertySpec.datatype` mapped to an XSD type, default `xsd:string`).

**GraphML/GEXF mapping.** Built via `networkx.MultiDiGraph`; both writers
require homogeneous attribute types across all nodes/edges sharing a key, so
every node/edge property value is stringified (JSON-encoded for lists/dicts)
before being handed to networkx -- documented data-fidelity trade-off: types
are recoverable by the consumer via the ontology's declared `datatype`, not by
the GraphML/GEXF attribute type.

**Cypher.** One `CREATE` per node (label = node type, a synthetic `_nid` key +
properties as map literal); one `MATCH ... CREATE` per edge (endpoints matched
on `_nid` on both sides -- no label assumed, so a dangling/edge-type ontology
mismatch never breaks the match). `_nid` is a dedicated identifier independent
of user properties, so an ontology with a property named `id` still matches
(a user `id` would otherwise shadow the lookup key and yield zero edges).
Deterministic: nodes/edges are sorted by `id`.

**Neo4j admin-import CSVs.** One `nodes_<Type>.csv` (header
`id:ID,:LABEL,<prop...>`) and one `edges_<Type>.csv` (header
`:START_ID,:END_ID,:TYPE,<prop...>`) per node/edge type -- properties differ
per type, so a single universal header would either drop columns or leave
them empty; per-type files match the real `neo4j-admin database import`
multi-file convention. Bundled as a small in-memory zip.

**GNN.** `.npz` carries `edge_index` (2, E) int64 (node-index space, in
`nodes` list order), a one-hot `node_features` (N, num_node_types) matrix (no
numeric-property tensorization in GC -- a documented follow-up), and the
`node_type_names`/`edge_type_names` label vocabularies needed to decode the
integer ids back to ontology type names. `graph-parquet` is the edge list
(`edge_id, source, target, type, directed`) written with pyarrow, for scale
(streams from a single `pyarrow.Table`, not row-by-row Python objects).
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
import zipfile
from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion, ExportArtifact
from anodyne_dataset.ports import Exporter
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace
from rdflib.namespace import XSD

from anodyne_graph.errors import UnsupportedGraphExportFormatError
from anodyne_graph.models import Edge, GraphDataset, GraphOntology, Node
from anodyne_graph.serialization import from_json_bytes, to_json_bytes

EX = Namespace("https://anodyne.ai/graph#")
ONTO = Namespace("https://anodyne.ai/onto#")

# The RDF-serialization formats GC supports (rdflib plugin name each maps to).
_RDF_FORMAT_NAMES = {"ttl": "turtle", "nt": "nt", "jsonld": "json-ld", "rdfxml": "xml"}

GRAPH_SUPPORTED_FORMATS = frozenset(
    {"graph_json", "owl", "graphml", "gexf", "cypher", "neo4j-csv", "npz", "graph-parquet"}
    | set(_RDF_FORMAT_NAMES)
)

# Object-key file extension per format (mirrors `anodyne_export.exporter._EXTENSIONS`'s role).
_GRAPH_EXTENSIONS = {
    "graph_json": "json",
    "ttl": "ttl",
    "nt": "nt",
    "jsonld": "jsonld",
    "rdfxml": "rdf",
    "owl": "owl",
    "graphml": "graphml",
    "gexf": "gexf",
    "cypher": "cypher",
    "neo4j-csv": "zip",
    "npz": "npz",
    "graph-parquet": "parquet",
}

_DATATYPE_TO_XSD = {
    "string": XSD.string,
    "integer": XSD.integer,
    "float": XSD.double,
    "boolean": XSD.boolean,
    "datetime": XSD.dateTime,
}

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _local(value: str) -> str:
    """Sanitize a node/edge/type id for use as a URI local name or filename
    component: anything other than word chars/dash/dot collapses to `_`."""
    cleaned = _UNSAFE_ID_CHARS.sub("_", str(value))
    return cleaned or "_"


def _literal(value: Any, declared: str | None = None) -> Literal:
    # When the ontology declares the property `datetime`, type the A-Box literal
    # `xsd:dateTime` so it agrees with the OWL T-Box `rdfs:range` (which
    # `ontology_to_owl` emits as `xsd:dateTime`). datetime values arrive as ISO
    # strings (JSON has no datetime type), so without this they'd serialize as
    # plain string literals and disagree with the declared range. Other datatypes
    # already agree: int->xsd:integer, float->xsd:double, bool->xsd:boolean, and
    # a plain literal is xsd:string per RDF 1.1.
    if declared == "datetime" and isinstance(value, str):
        return Literal(value, datatype=XSD.dateTime)
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.double)
    return Literal(str(value))


def dataset_to_rdf(dataset: GraphDataset) -> Graph:
    """Instance data (A-Box) -> an RDF graph (see module docstring for the mapping)."""
    g = Graph()
    g.bind("ex", EX)
    g.bind("onto", ONTO)
    # Declared property datatypes, per type, so A-Box literals can be typed to
    # match the OWL T-Box range (see `_literal`).
    onto = dataset.ontology
    node_prop_dt = {
        nt.name: {p.name: p.datatype for p in nt.properties} for nt in onto.node_types
    }
    edge_prop_dt = {
        et.name: {p.name: p.datatype for p in et.properties} for et in onto.edge_types
    }
    for node in sorted(dataset.nodes, key=lambda n: n.id):
        subj = EX[_local(node.id)]
        g.add((subj, RDF.type, ONTO[_local(node.type)]))
        declared = node_prop_dt.get(node.type, {})
        for key, value in sorted(node.properties.items()):
            g.add((subj, ONTO[_local(key)], _literal(value, declared.get(key))))
    for edge in sorted(dataset.edges, key=lambda e: e.id):
        s = EX[_local(edge.source)]
        o = EX[_local(edge.target)]
        p = ONTO[_local(edge.type)]
        g.add((s, p, o))
        if edge.properties:
            # Standard reification (not RDF-star) -- see module docstring.
            # Deterministic bnode id (not a random `BNode()`) for stable output.
            stmt = BNode(f"stmt-{_local(edge.id)}")
            g.add((stmt, RDF.type, RDF.Statement))
            g.add((stmt, RDF.subject, s))
            g.add((stmt, RDF.predicate, p))
            g.add((stmt, RDF.object, o))
            edge_declared = edge_prop_dt.get(edge.type, {})
            for key, value in sorted(edge.properties.items()):
                g.add((stmt, ONTO[_local(key)], _literal(value, edge_declared.get(key))))
    return g


def ontology_to_owl(ontology: GraphOntology) -> Graph:
    """T-Box -> an OWL ontology graph (see module docstring for the mapping)."""
    g = Graph()
    g.bind("owl", OWL)
    g.bind("onto", ONTO)
    for nt in sorted(ontology.node_types, key=lambda n: n.name):
        cls = ONTO[_local(nt.name)]
        g.add((cls, RDF.type, OWL.Class))
        parent = ontology.subclass_of.get(nt.name)
        if parent:
            g.add((cls, RDFS.subClassOf, ONTO[_local(parent)]))
        for prop in nt.properties:
            prop_uri = ONTO[_local(f"{nt.name}.{prop.name}")]
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.domain, cls))
            g.add((prop_uri, RDFS.range, _DATATYPE_TO_XSD.get(prop.datatype, XSD.string)))
    for et in sorted(ontology.edge_types, key=lambda e: e.name):
        prop_uri = ONTO[_local(et.name)]
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.domain, ONTO[_local(et.source_type)]))
        g.add((prop_uri, RDFS.range, ONTO[_local(et.target_type)]))
        for prop in et.properties:
            eprop_uri = ONTO[_local(f"{et.name}.{prop.name}")]
            g.add((eprop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((eprop_uri, RDFS.domain, prop_uri))
            g.add((eprop_uri, RDFS.range, _DATATYPE_TO_XSD.get(prop.datatype, XSD.string)))
    return g


def _rdf_bytes(dataset: GraphDataset, fmt: str) -> bytes:
    text = dataset_to_rdf(dataset).serialize(format=_RDF_FORMAT_NAMES[fmt])
    return text.encode("utf-8")


def _owl_bytes(ontology: GraphOntology) -> bytes:
    text = ontology_to_owl(ontology).serialize(format="xml")
    return text.encode("utf-8")


def _stringify_props(props: dict[str, Any]) -> dict[str, str]:
    # GraphML/GEXF require a homogeneous attribute type across all nodes/edges
    # sharing a key -- stringify (JSON-encode compound values) so heterogeneous
    # ontologies never trip the writer. `None` is dropped (nullable property
    # absent) rather than serialized as the string "None".
    out: dict[str, str] = {}
    for key, value in sorted(props.items()):
        if value is None:
            continue
        out[key] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    return out


def _edge_directed(dataset: GraphDataset, edge: Edge) -> bool:
    # `directed` is declared on the ontology's `EdgeType`, not on the `Edge`
    # instance itself (see `anodyne_graph.models`); default True for an
    # instance whose type isn't in the ontology (a malformed artifact).
    edge_type = dataset.ontology.edge_type(edge.type)
    return edge_type.directed if edge_type is not None else True


def dataset_to_networkx(dataset: GraphDataset) -> nx.MultiDiGraph[str]:
    # Attributes are assigned via `.update()` rather than splatted as kwargs into
    # `add_node`/`add_edge`: a property literally named `type`, `key`,
    # `directed`, `u_of_edge`, ... would otherwise collide with a networkx
    # reserved keyword and raise `TypeError`. Structural attrs (`type`,
    # `directed`) are written last so they win over any same-named property.
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    for node in dataset.nodes:
        g.add_node(node.id)
        g.nodes[node.id].update({**_stringify_props(node.properties), "type": node.type})
    for edge in dataset.edges:
        key = g.add_edge(edge.source, edge.target, key=edge.id)
        g.edges[edge.source, edge.target, key].update(
            {
                **_stringify_props(edge.properties),
                "type": edge.type,
                "directed": str(_edge_directed(dataset, edge)),
            }
        )
    return g


def _graphml_bytes(dataset: GraphDataset) -> bytes:
    buf = io.BytesIO()
    nx.write_graphml(dataset_to_networkx(dataset), buf, encoding="utf-8")
    return buf.getvalue()


def _gexf_bytes(dataset: GraphDataset) -> bytes:
    buf = io.BytesIO()
    nx.write_gexf(dataset_to_networkx(dataset), buf, encoding="utf-8")
    return buf.getvalue()


def _cypher_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    # Escape control characters too: a raw newline/tab/CR inside a Cypher string
    # literal is rejected by some drivers. Backslashes are already doubled above,
    # so these produce two-char escape sequences; any remaining C0 char becomes
    # a `\uXXXX` escape.
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    escaped = "".join(c if ord(c) >= 0x20 else f"\\u{ord(c):04x}" for c in escaped)
    return f'"{escaped}"'


def _cypher_props(props: dict[str, Any]) -> str:
    body = ", ".join(f"{_local(k)}: {_cypher_value(v)}" for k, v in sorted(props.items()))
    return "{" + body + "}"


def dataset_to_cypher(dataset: GraphDataset) -> bytes:
    """A Cypher script: one `CREATE` per node, one `MATCH ... CREATE` per edge.

    Endpoints are matched on a dedicated synthetic key `_nid` (the graph's own
    `node.id`), NOT on a user property named `id`: a real `id` property would
    otherwise clobber the lookup key so every edge `MATCH` finds nothing and
    zero relationships are created. `_nid` is written last so a user property
    literally named `_nid` cannot override it, and any user `id`/`_eid`
    properties are preserved verbatim.
    """
    lines: list[str] = []
    for node in sorted(dataset.nodes, key=lambda n: n.id):
        props = _cypher_props({**node.properties, "_nid": node.id})
        lines.append(f"CREATE (:{_local(node.type)} {props});")
    for edge in sorted(dataset.edges, key=lambda e: e.id):
        rel_props = _cypher_props({**edge.properties, "_eid": edge.id})
        lines.append(
            f"MATCH (a {{_nid: {_cypher_value(edge.source)}}}), "
            f"(b {{_nid: {_cypher_value(edge.target)}}}) "
            f"CREATE (a)-[:{_local(edge.type)} {rel_props}]->(b);"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_csv_rows(header: list[str], rows: list[list[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def _group_by_type(items: list[Node] | list[Edge]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for item in items:
        grouped.setdefault(item.type, []).append(item)
    return grouped


def dataset_to_neo4j_csv_zip(dataset: GraphDataset) -> bytes:
    """Neo4j admin-import bundle: `nodes_<Type>.csv` + `edges_<Type>.csv`
    (one pair of files per type -- see module docstring), zipped in-memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        nodes_by_type = _group_by_type(dataset.nodes)
        for ntype in sorted(nodes_by_type):
            nodes: list[Node] = sorted(nodes_by_type[ntype], key=lambda n: n.id)
            prop_keys = sorted({k for n in nodes for k in n.properties})
            rows = [[n.id, n.type, *[n.properties.get(k, "") for k in prop_keys]] for n in nodes]
            csv_text = _write_csv_rows(["id:ID", ":LABEL", *prop_keys], rows)
            zf.writestr(f"nodes_{_local(ntype)}.csv", csv_text)

        edges_by_type = _group_by_type(dataset.edges)
        for etype in sorted(edges_by_type):
            edges: list[Edge] = sorted(edges_by_type[etype], key=lambda e: e.id)
            eprop_keys = sorted({k for e in edges for k in e.properties})
            erows = [
                [e.source, e.target, e.type, *[e.properties.get(k, "") for k in eprop_keys]]
                for e in edges
            ]
            csv_text = _write_csv_rows([":START_ID", ":END_ID", ":TYPE", *eprop_keys], erows)
            zf.writestr(f"edges_{_local(etype)}.csv", csv_text)
    return buf.getvalue()


def dataset_to_npz(dataset: GraphDataset) -> bytes:
    """GNN arrays: `edge_index` (2, E) + a one-hot `node_features` (N, T)
    matrix, plus the type-name vocabularies needed to decode ids -- see module
    docstring."""
    node_ids = [n.id for n in dataset.nodes]
    index_of = {nid: i for i, nid in enumerate(node_ids)}
    node_type_names = sorted({n.type for n in dataset.nodes})
    type_index = {t: i for i, t in enumerate(node_type_names)}
    num_types = max(len(node_type_names), 1)

    node_type_ids = np.array([type_index[n.type] for n in dataset.nodes], dtype=np.int64)
    node_features = np.zeros((len(dataset.nodes), num_types), dtype=np.float32)
    for i, node in enumerate(dataset.nodes):
        node_features[i, type_index[node.type]] = 1.0

    # Edges whose endpoints aren't in `nodes` (a malformed artifact) are
    # dropped rather than raising -- GC exports best-effort structure.
    valid_edges = [e for e in dataset.edges if e.source in index_of and e.target in index_of]
    edge_type_names = sorted({e.type for e in valid_edges})
    edge_type_index = {t: i for i, t in enumerate(edge_type_names)}
    if valid_edges:
        edge_index = np.array(
            [[index_of[e.source] for e in valid_edges], [index_of[e.target] for e in valid_edges]],
            dtype=np.int64,
        )
        edge_type_ids = np.array([edge_type_index[e.type] for e in valid_edges], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type_ids = np.zeros((0,), dtype=np.int64)

    buf = io.BytesIO()
    np.savez(
        buf,
        node_ids=np.array(node_ids, dtype=object),
        node_type_names=np.array(node_type_names, dtype=object),
        node_type_ids=node_type_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_type_names=np.array(edge_type_names, dtype=object),
        edge_type_ids=edge_type_ids,
    )
    return buf.getvalue()


def dataset_to_graph_parquet(dataset: GraphDataset) -> bytes:
    """The edge list (`edge_id, source, target, type, directed`) as a single
    Parquet table -- adjacency-at-scale per the module docstring."""
    table = pa.table(
        {
            "edge_id": [e.id for e in dataset.edges],
            "source": [e.source for e in dataset.edges],
            "target": [e.target for e in dataset.edges],
            "type": [e.type for e in dataset.edges],
            "directed": [_edge_directed(dataset, e) for e in dataset.edges],
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def encode_dataset(dataset: GraphDataset, fmt: str) -> bytes:
    """Serialize `dataset` to `fmt` (one of `GRAPH_SUPPORTED_FORMATS`).

    Raises `UnsupportedGraphExportFormatError` for any other format string.
    """
    if fmt == "graph_json":
        return to_json_bytes(dataset)
    if fmt in _RDF_FORMAT_NAMES:
        return _rdf_bytes(dataset, fmt)
    if fmt == "owl":
        return _owl_bytes(dataset.ontology)
    if fmt == "graphml":
        return _graphml_bytes(dataset)
    if fmt == "gexf":
        return _gexf_bytes(dataset)
    if fmt == "cypher":
        return dataset_to_cypher(dataset)
    if fmt == "neo4j-csv":
        return dataset_to_neo4j_csv_zip(dataset)
    if fmt == "npz":
        return dataset_to_npz(dataset)
    if fmt == "graph-parquet":
        return dataset_to_graph_parquet(dataset)
    raise UnsupportedGraphExportFormatError(
        f"unsupported graph export format {fmt!r}; expected one of "
        f"{sorted(GRAPH_SUPPORTED_FORMATS)}"
    )


def _artifact_key(dataset_id: uuid.UUID, version_id: uuid.UUID, ext: str) -> str:
    # Same tenant-relative convention as `anodyne_export.exporter._artifact_key`.
    return f"datasets/{dataset_id}/{version_id}/export.{ext}"


class GraphExporter(Exporter):
    """The `Exporter` adapter for `graph_json` artifacts.

    Loads the artifact once (`from_json_bytes`), encodes it to the requested
    format, and uploads it -- exactly the same shape as
    `anodyne_export.exporter.PyArrowExporter` so `export_routes.py` can treat
    the two exporters interchangeably behind the `Exporter` port.
    """

    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = 50_000,
    ) -> ExportArtifact:
        resolved = format or "graph_json"
        if resolved not in GRAPH_SUPPORTED_FORMATS:
            raise UnsupportedGraphExportFormatError(
                f"unsupported graph export format {resolved!r}; expected one of "
                f"{sorted(GRAPH_SUPPORTED_FORMATS)}"
            )

        data = await store.get(version.artifact_uri)
        dataset = from_json_bytes(data)

        encoded = encode_dataset(dataset, resolved)
        key = _artifact_key(version.dataset_id, version.id, _GRAPH_EXTENSIONS[resolved])
        await store.put(key, encoded)

        return ExportArtifact(
            id=uuid.uuid4(),
            tenant_id=version.tenant_id,
            dataset_id=version.dataset_id,
            version_id=version.id,
            format=resolved,
            row_count=len(dataset.nodes) + len(dataset.edges),
            object_key=key,
        )
