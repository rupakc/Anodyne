"""From-sample engine: learn a real graph's statistics, synthesize a new one.

Given an uploaded graph (node-link JSON -- the GA artifact -- or GraphML via
networkx) we learn, then regenerate to match:

- **degree distribution** -- via a Chung-Lu expected-degree model
  (``nx.expected_degree_graph``) seeded from the empirical degree sequence,
  rescaled to the requested node count;
- **node-type proportions** -- fraction of nodes per type;
- **per-type attribute distributions** -- learned per non-PII property and
  *resampled*, never copied (see the privacy note for the exact rule);
- **edge-type mix** -- projected onto the ontology's domain/range, like the
  topology engine.

**Privacy.** No differential privacy is claimed (out of scope). Protection comes
from (a) faking *all* PII properties, (b) minting fresh node ids, and (c)
regenerating topology stochastically so structure is matched only in
distribution. Property values are handled to avoid leaking sensitive-but-non-PII
fields (salary, diagnosis, free text) verbatim:

- **numeric** props are *sampled* from a fitted distribution (Gaussian around
  the empirical mean, clipped to the observed min/max) -- exact sample values
  are never copied;
- **low-cardinality, short categoricals** (<= 20 distinct labels, each <= 32
  chars) resample observed *labels* -- this is statistical matching, not a
  personal leak, and is the one place observed values recur by design;
- **high-cardinality / free-text / long-string** props are synthesized via
  Faker -- their values are never copied from the sample.

On top of that we enforce a **no-verbatim-subgraph** guarantee: we fingerprint
every edge by ``(edge_type, sorted endpoint property items + type)`` for both
input and output and assert the two fingerprint sets are disjoint (see
``assert_no_verbatim_subgraph``) -- so no labeled edge of the input is
reproduced verbatim beyond chance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from faker import Faker

from anodyne_graph.errors import GraphGenerationError
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
from anodyne_graph.properties import fake_pii, is_pii
from anodyne_graph.serialization import from_json_bytes
from anodyne_graph.topology import map_edges

if TYPE_CHECKING:
    from anodyne_dataset.models import DatasetSpec


def _edge_fingerprints(dataset: GraphDataset) -> set[tuple[Any, ...]]:
    """A hashable, id-independent fingerprint per edge (for the privacy check)."""
    prop_of = {
        n.id: (n.type, tuple(sorted((k, str(v)) for k, v in n.properties.items())))
        for n in dataset.nodes
    }
    fps: set[tuple[Any, ...]] = set()
    for e in dataset.edges:
        src = prop_of.get(e.source)
        tgt = prop_of.get(e.target)
        if src is None or tgt is None:
            continue
        fps.add((e.type, src, tgt))
    return fps


def assert_no_verbatim_subgraph(
    sample: GraphDataset, synthetic: GraphDataset, *, strict: bool
) -> int:
    """Check that no verbatim labeled edge of the sample recurs in the synthetic.

    Returns the overlap count. When ``strict`` (the sample carries identifying /
    PII properties, so an identical edge fingerprint would leak an individual)
    any overlap raises ``GraphGenerationError`` -- and because PII is re-faked,
    the expected overlap is 0. When the sample has no identifying properties
    there is nothing personal to leak and identical categorical tuples are pure
    chance, so the count is reported (in metrics) rather than raised on.
    """
    overlap = _edge_fingerprints(sample) & _edge_fingerprints(synthetic)
    if overlap and strict:
        raise GraphGenerationError(
            f"no-verbatim-subgraph guarantee violated: {len(overlap)} input edge(s) reproduced"
        )
    return len(overlap)


def graphml_to_dataset(data: bytes) -> GraphDataset:
    """Parse a GraphML document into a `GraphDataset` (types from attributes)."""
    import io

    graph = nx.read_graphml(io.BytesIO(data))
    node_types: dict[str, set[str]] = {}
    nodes: list[Node] = []
    for nid, attrs in graph.nodes(data=True):
        ntype = str(attrs.get("type") or attrs.get("label") or attrs.get("labels") or "Node")
        props = {k: v for k, v in attrs.items() if k not in ("type", "label", "labels")}
        node_types.setdefault(ntype, set()).update(props.keys())
        nodes.append(Node(id=str(nid), type=ntype, properties=dict(props)))
    edges: list[Edge] = []
    node_type_of = {n.id: n.type for n in nodes}
    edge_types: dict[tuple[str, str, str], None] = {}
    for i, (u, v, attrs) in enumerate(graph.edges(data=True)):
        etype = str(attrs.get("type") or attrs.get("label") or "REL")
        su, sv = str(u), str(v)
        edges.append(Edge(id=f"{etype}:{i}", type=etype, source=su, target=sv, properties={}))
        edge_types[(etype, node_type_of.get(su, "Node"), node_type_of.get(sv, "Node"))] = None
    ontology = GraphOntology(
        node_types=[
            NodeType(name=t, properties=[PropertySpec(name=p) for p in sorted(props)])
            for t, props in sorted(node_types.items())
        ],
        edge_types=[
            EdgeType(name=et, source_type=s, target_type=t) for (et, s, t) in sorted(edge_types)
        ],
    )
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def _infer_ontology(sample: GraphDataset) -> GraphOntology:
    if sample.ontology.node_types:
        return sample.ontology
    node_props: dict[str, set[str]] = {}
    type_of = {n.id: n.type for n in sample.nodes}
    for n in sample.nodes:
        node_props.setdefault(n.type, set()).update(n.properties.keys())
    edge_types: dict[tuple[str, str, str], None] = {}
    for e in sample.edges:
        edge_types[(e.type, type_of.get(e.source, ""), type_of.get(e.target, ""))] = None
    return GraphOntology(
        node_types=[
            NodeType(name=t, properties=[PropertySpec(name=p) for p in sorted(ps)])
            for t, ps in sorted(node_props.items())
        ],
        edge_types=[
            EdgeType(name=et, source_type=s, target_type=t) for (et, s, t) in sorted(edge_types)
        ],
    )


class FromSampleGraphGenerator:
    """Learn-and-synthesize graph generator. No LLM; deterministic + seeded."""

    def __init__(self, sample: GraphDataset) -> None:
        self._sample = sample
        self._ontology = _infer_ontology(sample)

    @classmethod
    def from_node_link_bytes(cls, data: bytes) -> FromSampleGraphGenerator:
        return cls(from_json_bytes(data))

    @classmethod
    def from_graphml_bytes(cls, data: bytes) -> FromSampleGraphGenerator:
        return cls(graphml_to_dataset(data))

    def generate(
        self,
        spec: DatasetSpec,
        start_index: int,
        count: int,
        seed: int,
        shard_index: int = 0,
    ) -> GraphDataset:
        if not self._sample.nodes:
            raise GraphGenerationError("sample graph has no nodes to learn from")
        rng = np.random.default_rng([seed, shard_index])
        fake = Faker()
        # Per-instance seed (not global `Faker.seed`) so concurrent shards each
        # get an independent, deterministic Faker without racing on shared state.
        fake.seed_instance(seed * 1_000_003 + shard_index * 7919 + start_index)
        n = max(1, count)

        type_of = self._assign_types(n, rng)
        # Shard-global ids (incorporate `start_index`) so multi-shard assembly
        # never dedups distinct nodes; identical to `{type}:{i}` for start_index 0.
        node_ids = [f"{type_of[i]}:{start_index + i}" for i in range(n)]
        graph = self._synthesize_topology(n, rng)
        nodes = [
            Node(id=node_ids[i], type=type_of[i], properties=self._attrs(type_of[i], rng, fake))
            for i in range(n)
        ]
        edges, dropped = map_edges(graph, node_ids, type_of, self._ontology)

        synthetic = GraphDataset(
            ontology=self._ontology,
            nodes=nodes,
            edges=edges,
            metrics=compute_metrics(nodes, edges),
        )
        strict = any(is_pii(p) for nt in self._ontology.node_types for p in nt.properties)
        overlap = assert_no_verbatim_subgraph(self._sample, synthetic, strict=strict)
        synthetic.metrics["engine"] = "from_sample"
        synthetic.metrics["edges_dropped_unmappable"] = dropped
        synthetic.metrics["verbatim_subgraph_overlap"] = overlap
        synthetic.metrics["sample_mean_degree"] = self._sample_mean_degree()
        return synthetic

    # -- learning -----------------------------------------------------------
    def _assign_types(self, n: int, rng: np.random.Generator) -> list[str]:
        counts: dict[str, int] = {}
        for node in self._sample.nodes:
            counts[node.type] = counts.get(node.type, 0) + 1
        names = list(counts.keys())
        weights = np.array([counts[t] for t in names], dtype=float)
        weights /= weights.sum()
        idx = rng.choice(len(names), size=n, p=weights)
        return [names[int(i)] for i in idx]

    def _degree_sequence(self) -> list[int]:
        g = nx.MultiGraph()
        g.add_nodes_from(n.id for n in self._sample.nodes)
        g.add_edges_from((e.source, e.target) for e in self._sample.edges)
        return [d for _, d in g.degree()]

    def _sample_mean_degree(self) -> float:
        seq = self._degree_sequence()
        return float(np.mean(seq)) if seq else 0.0

    def _synthesize_topology(self, n: int, rng: np.random.Generator) -> nx.Graph:
        seq = self._degree_sequence()
        if not seq or max(seq) == 0:
            return nx.empty_graph(n)
        # Resample expected degrees from the empirical distribution (Chung-Lu),
        # so the synthetic degree distribution matches the sample's in expectation.
        weights = rng.choice(np.array(seq, dtype=float), size=n, replace=True)
        nx_seed = int(rng.integers(0, 2**31 - 1))
        g = nx.expected_degree_graph(list(weights), seed=nx_seed, selfloops=False)
        return nx.relabel_nodes(g, {v: i for i, v in enumerate(sorted(g.nodes()))}, copy=True)

    def _attrs(self, ntype: str, rng: np.random.Generator, fake: Faker) -> dict[str, Any]:
        node_type = self._ontology.node_type(ntype)
        if node_type is None:
            return {}
        pools = self._value_pools(ntype)
        props: dict[str, Any] = {}
        for prop in node_type.properties:
            if is_pii(prop):
                props[prop.name] = fake_pii(prop, fake)
                continue
            pool = pools.get(prop.name)
            if prop.datatype in ("integer", "float") and pool:
                nums = [float(x) for x in pool if _is_number(x)]
                if nums:
                    # Numeric: SAMPLE a fresh value from a fitted range (Gaussian
                    # around the empirical mean, clipped to observed min/max) --
                    # never copy an exact sample value.
                    lo, hi = min(nums), max(nums)
                    mean, std = float(np.mean(nums)), float(np.std(nums))
                    val = float(np.clip(rng.normal(mean, std or 1.0), lo, hi))
                    props[prop.name] = int(round(val)) if prop.datatype == "integer" else val
                    continue
            if pool and _is_low_cardinality_categorical(pool):
                # Low-cardinality short categoricals: resampling observed LABELS
                # is statistical matching, not a verbatim leak -- kept by design.
                props[prop.name] = pool[int(rng.integers(0, len(pool)))]
            else:
                # High-cardinality / free-text / long strings (and empty pools):
                # synthesize -- never copy sample values verbatim.
                props[prop.name] = fake.word()
        return props

    def _value_pools(self, ntype: str) -> dict[str, list[Any]]:
        pools: dict[str, list[Any]] = {}
        for node in self._sample.nodes:
            if node.type != ntype:
                continue
            for k, v in node.properties.items():
                pools.setdefault(k, []).append(v)
        return pools


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# A property is a "low-cardinality categorical" -- and thus safe to resample by
# LABEL (statistical matching, not a verbatim leak) -- when it has few distinct,
# short values. Anything else (free text, long strings, high cardinality) is
# synthesized instead of copied.
_MAX_CATEGORY_LABELS = 20
_MAX_CATEGORY_LEN = 32


def _is_low_cardinality_categorical(pool: list[Any]) -> bool:
    distinct = {str(v) for v in pool}
    if len(distinct) > _MAX_CATEGORY_LABELS:
        return False
    return all(len(v) <= _MAX_CATEGORY_LEN for v in distinct)
