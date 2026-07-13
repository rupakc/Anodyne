"""Graph-modality perturbations (Track GH): structural + semantic corruption.

Extends sub-system D to the graph modality. `perturb_graph` applies one
`PerturbationFamily` to a `GraphDataset`, returning a **new** dataset (the input
is never mutated). Three families:

- ``GRAPH_REWIRE`` — structural: rewire a fraction of edges while preserving the
  node set; ``params["degree_preserving"]`` keeps every node's in/out degree via
  double-edge (target) swaps.
- ``GRAPH_DROPOUT`` — structural: drop a fraction of edges and/or nodes
  (``params["target"]`` in ``{"edges","nodes","both"}``, default ``"edges"``);
  dropping a node removes its now-dangling edges so no edge references a missing
  endpoint.
- ``GRAPH_ONTOLOGY_VIOLATION`` — semantic: inject a controlled number of
  ontology violations (out-of-range / bad-choice / missing-required property
  values, undeclared node types, wrong edge endpoints) that the
  ``OntologyConsistencyGraphJudge`` measurably flags.

Determinism follows the repo idiom: a seeded ``np.random.default_rng([seed,
family_ordinal])`` chooses affected elements over **id-sorted** iteration, so the
same ``(dataset, family, intensity, seed, params)`` always yields byte-identical
output regardless of the input's node/edge ordering (which is preserved in the
result). ``intensity`` in ``[0, 1]`` scales the affected fraction; ``intensity``
0 is an exact no-op. Any newly written property value never touches a PII
property, so faked PII stays faked and no PII is ever synthesized here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from anodyne_dataset.models import PerturbationFamily

from anodyne_graph.models import (
    Edge,
    GraphDataset,
    GraphOntology,
    Node,
    PropertySpec,
    compute_metrics,
)
from anodyne_graph.properties import is_pii


def _family_ord(family: PerturbationFamily) -> int:
    return list(PerturbationFamily).index(family)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _n_affected(intensity: float, total: int) -> int:
    return int(round(_clamp01(intensity) * total))


def _select_indices(items: list[Node] | list[Edge], k: int, rng: np.random.Generator) -> set[int]:
    """Deterministically choose `k` original-position indices from `items`.

    Selection ranks candidates by id (stable regardless of input order) and
    draws `k` without replacement, so output is reproducible for a given seed.
    """
    if k <= 0 or not items:
        return set()
    order = sorted(range(len(items)), key=lambda i: items[i].id)
    k = min(k, len(order))
    chosen = rng.choice(len(order), size=k, replace=False)
    return {order[int(c)] for c in chosen}


def _finalize(ontology: GraphOntology, nodes: list[Node], edges: list[Edge]) -> GraphDataset:
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def perturb_graph(
    dataset: GraphDataset,
    family: PerturbationFamily,
    intensity: float,
    seed: int,
    params: dict[str, Any] | None = None,
) -> GraphDataset:
    """Apply a graph `family` to `dataset`, returning a corrupted copy.

    Never mutates `dataset`. Deterministic given `(family, intensity, seed,
    params)`. Raises `ValueError` for a non-graph family.
    """
    params = dict(params or {})
    ontology = dataset.ontology.model_copy(deep=True)
    nodes = [n.model_copy(deep=True) for n in dataset.nodes]
    edges = [e.model_copy(deep=True) for e in dataset.edges]
    rng = np.random.default_rng([seed, _family_ord(family)])

    if family is PerturbationFamily.GRAPH_REWIRE:
        edges = _rewire(nodes, edges, intensity, params, rng)
    elif family is PerturbationFamily.GRAPH_DROPOUT:
        nodes, edges = _dropout(nodes, edges, intensity, params, rng)
    elif family is PerturbationFamily.GRAPH_ONTOLOGY_VIOLATION:
        nodes, edges = _ontology_violation(ontology, nodes, edges, intensity, params, rng)
    else:
        raise ValueError(f"{family} is not a graph perturbation family")

    return _finalize(ontology, nodes, edges)


# --------------------------------------------------------------------------- #
# Structural: rewire
# --------------------------------------------------------------------------- #
def _rewire(
    nodes: list[Node],
    edges: list[Edge],
    intensity: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> list[Edge]:
    if not edges:
        return edges
    k = _n_affected(intensity, len(edges))
    if bool(params.get("degree_preserving")):
        return _degree_preserving_swaps(edges, k, rng)
    node_ids = sorted(n.id for n in nodes)
    for i in sorted(_select_indices(edges, k, rng)):
        e = edges[i]
        edges[i] = e.model_copy(
            update={"target": _pick_different(node_ids, e.source, e.target, rng)}
        )
    return edges


def _pick_different(
    node_ids: list[str], source: str, current: str, rng: np.random.Generator
) -> str:
    candidates = [x for x in node_ids if x != source and x != current]
    if not candidates:
        candidates = [x for x in node_ids if x != current]
    if not candidates:
        return current
    return candidates[int(rng.integers(0, len(candidates)))]


def _degree_preserving_swaps(edges: list[Edge], k: int, rng: np.random.Generator) -> list[Edge]:
    """Swap the targets of `k` edge pairs -- preserves every node's in/out degree."""
    m = len(edges)
    if m < 2 or k <= 0:
        return edges
    order = sorted(range(m), key=lambda i: edges[i].id)
    for _ in range(k):
        a, b = (int(x) for x in rng.choice(len(order), size=2, replace=False))
        ia, ib = order[a], order[b]
        ea, eb = edges[ia], edges[ib]
        edges[ia] = ea.model_copy(update={"target": eb.target})
        edges[ib] = eb.model_copy(update={"target": ea.target})
    return edges


# --------------------------------------------------------------------------- #
# Structural: dropout
# --------------------------------------------------------------------------- #
def _dropout(
    nodes: list[Node],
    edges: list[Edge],
    intensity: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[Node], list[Edge]]:
    target = str(params.get("target", "edges")).lower()
    do_nodes = target in ("nodes", "both")
    do_edges = target in ("edges", "both")
    if do_nodes:
        drop = _select_indices(nodes, _n_affected(intensity, len(nodes)), rng)
        nodes = [n for i, n in enumerate(nodes) if i not in drop]
        kept = {n.id for n in nodes}
        # Remove edges left dangling by a dropped endpoint.
        edges = [e for e in edges if e.source in kept and e.target in kept]
    if do_edges:
        drop = _select_indices(edges, _n_affected(intensity, len(edges)), rng)
        edges = [e for i, e in enumerate(edges) if i not in drop]
    return nodes, edges


# --------------------------------------------------------------------------- #
# Semantic: ontology-violation injection
# --------------------------------------------------------------------------- #
def _ontology_violation(
    ontology: GraphOntology,
    nodes: list[Node],
    edges: list[Edge],
    intensity: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[Node], list[Edge]]:
    for i in sorted(_select_indices(nodes, _n_affected(intensity, len(nodes)), rng)):
        nodes[i] = _violate_node(ontology, nodes[i])
    for i in sorted(_select_indices(edges, _n_affected(intensity, len(edges)), rng)):
        edges[i] = _violate_edge(ontology, edges[i], nodes, rng)
    return nodes, edges


def _range_violation(prop: PropertySpec) -> Any:
    hi = prop.constraints.get("max")
    lo = prop.constraints.get("min")
    if isinstance(hi, (int, float)) and not isinstance(hi, bool):
        return hi + 1000
    if isinstance(lo, (int, float)) and not isinstance(lo, bool):
        return lo - 1000
    return None


def _datatype_violation(prop: PropertySpec) -> Any:
    dt = prop.datatype.lower()
    if dt in ("integer", "float", "boolean"):
        return "__not_this_datatype__"
    if dt in ("string", "datetime"):
        return 1_234_567
    return None  # unknown/extended datatype: cannot guarantee a flagged violation


def _violate_node(ontology: GraphOntology, node: Node) -> Node:
    """Corrupt `node` so the ontology judge flags it, without touching PII.

    Tries the semantic violation kinds in a fixed order so a heterogeneous
    ontology exercises several (out-of-range, bad-choice, missing-required,
    wrong-datatype); a node type with no usable property falls back to an
    undeclared node type.
    """
    node_type = ontology.node_type(node.type)
    if node_type is not None:
        candidates = [
            p for p in sorted(node_type.properties, key=lambda p: p.name) if not is_pii(p)
        ]
        for prop in candidates:
            bad = _range_violation(prop)
            if bad is not None:
                return node.model_copy(update={"properties": {**node.properties, prop.name: bad}})
        for prop in candidates:
            choices = prop.constraints.get("choices")
            if isinstance(choices, list) and choices:
                return node.model_copy(
                    update={"properties": {**node.properties, prop.name: "__not_a_choice__"}}
                )
        for prop in candidates:
            if not prop.nullable:
                stripped = {k: v for k, v in node.properties.items() if k != prop.name}
                return node.model_copy(update={"properties": stripped})
        for prop in candidates:
            bad = _datatype_violation(prop)
            if bad is not None:
                return node.model_copy(update={"properties": {**node.properties, prop.name: bad}})
    return node.model_copy(update={"type": f"{node.type}__violation"})


def _violate_edge(
    ontology: GraphOntology, edge: Edge, nodes: list[Node], rng: np.random.Generator
) -> Edge:
    """Repoint `edge` to a domain/range-violating endpoint (or a missing one)."""
    edge_type = ontology.edge_type(edge.type)
    if edge_type is not None:
        wrong = sorted(n.id for n in nodes if n.type != edge_type.target_type)
        if wrong:
            return edge.model_copy(update={"target": wrong[int(rng.integers(0, len(wrong)))]})
    return edge.model_copy(update={"target": "__missing_endpoint__"})
