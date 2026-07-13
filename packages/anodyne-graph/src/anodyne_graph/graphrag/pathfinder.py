"""Deterministic multi-hop path sampler over a ``GraphDataset``.

Builds a sorted adjacency from the instance edges (undirected edge types are
walkable in both directions) and does seeded random walks that never revisit a
node or reuse an edge, so the sampled paths are simple (no trivial cycles) and
every hop is a real graph edge. Identical ``seed`` + graph => identical paths.
"""

from __future__ import annotations

import numpy as np

from anodyne_graph.errors import GraphRAGError
from anodyne_graph.graphrag.models import QAPath
from anodyne_graph.models import GraphDataset

MIN_HOPS = 2
MAX_HOPS = 4


def _directed_of(dataset: GraphDataset) -> dict[str, bool]:
    """Map edge-type name -> directed flag (default True for unknown types)."""
    return {et.name: et.directed for et in dataset.ontology.edge_types}


def build_adjacency(dataset: GraphDataset) -> dict[str, list[tuple[str, str]]]:
    """node_id -> sorted list of ``(edge_id, neighbor_id)`` traversable hops.

    Directed edges contribute only ``source -> target``; undirected edge types
    contribute both orientations. Edges whose endpoints are not in the node set
    are skipped. The per-node lists are de-duplicated and sorted for
    determinism.
    """
    directed = _directed_of(dataset)
    node_ids = {n.id for n in dataset.nodes}
    adj: dict[str, list[tuple[str, str]]] = {n.id: [] for n in dataset.nodes}
    for edge in dataset.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        adj[edge.source].append((edge.id, edge.target))
        if not directed.get(edge.type, True):
            adj[edge.target].append((edge.id, edge.source))
    return {node_id: sorted(set(hops)) for node_id, hops in adj.items()}


def _walk(
    adj: dict[str, list[tuple[str, str]]],
    start: str,
    target_len: int,
    min_hops: int,
    rng: np.random.Generator,
) -> QAPath | None:
    """One seeded simple walk from ``start`` of up to ``target_len`` hops."""
    visited = {start}
    used_edges: set[str] = set()
    current = start
    hops: list[tuple[str, str]] = []
    for _ in range(target_len):
        candidates = [
            (edge_id, nbr)
            for edge_id, nbr in adj[current]
            if nbr not in visited and edge_id not in used_edges
        ]
        if not candidates:
            break
        edge_id, nbr = candidates[int(rng.integers(0, len(candidates)))]
        hops.append((current, edge_id))
        used_edges.add(edge_id)
        visited.add(nbr)
        current = nbr
    if len(hops) < min_hops:
        return None
    return QAPath(hops=hops, terminal_node_id=current)


def sample_paths(
    dataset: GraphDataset,
    *,
    count: int,
    seed: int,
    min_hops: int = MIN_HOPS,
    max_hops: int = MAX_HOPS,
) -> list[QAPath]:
    """Sample up to ``count`` distinct simple multi-hop paths.

    Deterministic given ``seed``. Returns fewer than ``count`` paths only when
    the graph cannot yield more distinct ones.

    Raises:
        GraphRAGError: if the bounds are invalid, the graph is too small for the
            requested hop count, or no multi-hop path can be sampled at all.
    """
    if count < 1:
        raise GraphRAGError(f"count must be >= 1, got {count}")
    if min_hops < 1 or max_hops < min_hops:
        raise GraphRAGError(f"invalid hop bounds: min_hops={min_hops}, max_hops={max_hops}")
    if len(dataset.nodes) < min_hops + 1 or len(dataset.edges) < min_hops:
        raise GraphRAGError(
            f"graph too small to sample {min_hops}-hop paths: "
            f"{len(dataset.nodes)} nodes, {len(dataset.edges)} edges "
            f"(need >= {min_hops + 1} nodes and >= {min_hops} edges)"
        )

    adj = build_adjacency(dataset)
    node_ids = sorted(adj)
    rng = np.random.default_rng(seed)
    paths: list[QAPath] = []
    seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()
    max_attempts = max(count * 50, 200)
    for _ in range(max_attempts):
        if len(paths) >= count:
            break
        start = node_ids[int(rng.integers(0, len(node_ids)))]
        target_len = int(rng.integers(min_hops, max_hops + 1))
        path = _walk(adj, start, target_len, min_hops, rng)
        if path is None:
            continue
        key = (tuple(path.hops), path.terminal_node_id)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)

    if not paths:
        raise GraphRAGError(
            "could not sample any multi-hop path; the graph may be too sparse or fully disconnected"
        )
    return paths
