"""Procedural topology engine: seeded random-graph models mapped to an ontology.

Structure comes from ``networkx`` classic generators (no LLM); it is then
projected onto the ontology's node/edge types respecting domain/range. Selected
via ``directives["topology"]``; per-model params ride in ``directives``:

- ``barabasi_albert``   scale-free / power-law degree     -- ``m``
- ``watts_strogatz``    small-world                       -- ``k``, ``p``
- ``erdos_renyi``       Gilbert random graph              -- ``p``
- ``stochastic_block_model`` community structure    -- ``blocks``/``sizes``, ``p_in``, ``p_out``
- ``lfr``               LFR benchmark communities         -- ``mu``, ``average_degree``, ...

Determinism: ``np.random.default_rng([seed, shard_index])`` seeds a per-call
integer handed to networkx, plus a seeded Faker for node properties. Same seed
=> byte-identical graph.

Node-type projection: for community models (SBM/LFR) each community maps to a
node type (round-robin over the ontology's node types), so blocks become typed
sub-populations; otherwise node types are assigned round-robin. Each undirected
topology edge is oriented onto the first ontology edge type whose declared
``source_type -> target_type`` matches the endpoints' assigned types (trying the
reverse orientation too); edges with no compatible edge type are dropped and
counted in ``metrics["edges_dropped_unmappable"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from faker import Faker

from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.models import Edge, GraphDataset, GraphOntology, Node, compute_metrics
from anodyne_graph.properties import ontology_from_spec, synthesize_node_properties

if TYPE_CHECKING:
    from anodyne_dataset.models import DatasetSpec

TOPOLOGY_MODELS = frozenset(
    {"barabasi_albert", "watts_strogatz", "erdos_renyi", "stochastic_block_model", "lfr"}
)


def _nx_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _sbm_sizes(directives: dict[str, Any], n: int) -> list[int]:
    raw = directives.get("blocks") or directives.get("sizes")
    if isinstance(raw, list) and raw and all(isinstance(x, int) and x > 0 for x in raw):
        return [int(x) for x in raw]
    n_blocks = int(directives.get("n_blocks", 3))
    n_blocks = max(1, min(n_blocks, n))
    base, extra = divmod(n, n_blocks)
    return [base + (1 if i < extra else 0) for i in range(n_blocks)]


def build_topology(
    name: str, n: int, directives: dict[str, Any], rng: np.random.Generator
) -> tuple[nx.Graph, dict[int, int] | None]:
    """Build the abstract topology graph + an optional node->community map.

    Nodes are integers ``0..n-1``. The community map is populated for the
    community models (SBM/LFR) and ``None`` otherwise.

    Raises:
        GraphGenerationError: for an unknown model or params that networkx
            cannot satisfy (e.g. LFR failing to converge).
    """
    if n < 1:
        raise GraphGenerationError("topology needs at least 1 node")
    seed = _nx_seed(rng)
    if name == "barabasi_albert":
        m = max(1, min(int(directives.get("m", 2)), n - 1)) if n > 1 else 1
        return (nx.barabasi_albert_graph(n, m, seed=seed) if n > 1 else nx.empty_graph(n)), None
    if name == "watts_strogatz":
        k = int(directives.get("k", 4))
        k = max(2, min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)) if n > 2 else 2
        p = float(directives.get("p", 0.1))
        return nx.watts_strogatz_graph(n, min(k, n - 1), p, seed=seed), None
    if name == "erdos_renyi":
        p = float(directives.get("p", min(0.1, 10.0 / max(n, 1))))
        return nx.erdos_renyi_graph(n, p, seed=seed), None
    if name == "stochastic_block_model":
        sizes = _sbm_sizes(directives, n)
        p_in = float(directives.get("p_in", 0.3))
        p_out = float(directives.get("p_out", 0.02))
        b = len(sizes)
        probs = [[p_in if i == j else p_out for j in range(b)] for i in range(b)]
        g: nx.Graph = nx.stochastic_block_model(sizes, probs, seed=seed)
        communities = {int(v): int(g.nodes[v]["block"]) for v in g}
        return g, communities
    if name == "lfr":
        return _build_lfr(n, directives, rng)
    raise GraphGenerationError(
        f"unknown topology model {name!r}; expected one of {sorted(TOPOLOGY_MODELS)}"
    )


def _lfr_worker(
    q: Any, n: int, tau1: float, tau2: float, mu: float, avg: float, min_comm: int, seed: int
) -> None:  # pragma: no cover - runs in a child process
    """Child-process entry point: build one LFR graph and ship edges+communities."""
    import networkx as _nx

    try:
        g = _nx.LFR_benchmark_graph(
            n, tau1, tau2, mu, average_degree=avg, min_community=min_comm, max_iters=500, seed=seed
        )
        g.remove_edges_from(_nx.selfloop_edges(g))
        edges = [(int(u), int(v)) for u, v in g.edges()]
        comms = {int(v): tuple(sorted(int(x) for x in g.nodes[v]["community"])) for v in g}
        q.put(("ok", edges, comms))
    except Exception as exc:  # noqa: BLE001 - report any failure back to the parent
        q.put(("err", type(exc).__name__))


def _build_lfr(
    n: int, directives: dict[str, Any], rng: np.random.Generator
) -> tuple[nx.Graph, dict[int, int]]:
    """Build an LFR benchmark graph.

    ``networkx``'s LFR generator is known to *hang* (ignoring ``max_iters``) for
    some seeds, so each attempt runs in a killable subprocess with a wall-clock
    timeout; a hung/failed attempt is abandoned and the next deterministic seed
    is tried. The seed sequence is a deterministic function of ``rng``, so which
    attempt succeeds -- and thus the resulting graph -- is reproducible.
    """
    import multiprocessing as mp
    import queue

    tau1 = float(directives.get("tau1", 3.0))
    tau2 = float(directives.get("tau2", 1.5))
    mu = float(directives.get("mu", 0.1))
    avg_deg = float(directives.get("average_degree", 5))
    min_comm = int(directives.get("min_community", max(10, n // 10)))
    timeout_s = float(directives.get("lfr_timeout_s", 6.0))
    attempts = int(directives.get("lfr_attempts", 16))

    ctx = mp.get_context("spawn")
    for _ in range(attempts):
        seed = _nx_seed(rng)
        q: Any = ctx.Queue()
        proc = ctx.Process(target=_lfr_worker, args=(q, n, tau1, tau2, mu, avg_deg, min_comm, seed))
        proc.start()
        # Read the result FIRST: a large valid payload can exceed the OS pipe
        # buffer, blocking the child on `q.put` until the parent drains it. If
        # we `join` before `get`, that child would look hung and we'd falsely
        # declare non-convergence. `q.get` unblocks the child; `Empty` after
        # `timeout_s` is the real hang.
        try:
            tag, *rest = q.get(timeout=timeout_s)
        except queue.Empty:  # pragma: no cover - the hang path (RNG-dependent)
            proc.terminate()
            proc.join()
            q.close()
            q.join_thread()
            continue
        proc.join(timeout_s)
        if proc.is_alive():  # pragma: no cover - defensive: child not exiting post-put
            proc.terminate()
            proc.join()
        q.close()
        q.join_thread()
        if tag != "ok":
            continue
        edges, comm_tuples = rest
        g: nx.Graph = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        uniq: dict[tuple[int, ...], int] = {}
        communities: dict[int, int] = {}
        for v, ct in comm_tuples.items():
            communities[int(v)] = uniq.setdefault(tuple(ct), len(uniq))
        return g, communities
    raise GraphGenerationError(  # pragma: no cover - only if every seed hangs/fails
        f"LFR benchmark did not converge for n={n} in {attempts} attempts; "
        "adjust mu/average_degree/min_community"
    )


def assign_node_types(
    n: int,
    ontology: GraphOntology,
    communities: dict[int, int] | None,
) -> list[str]:
    """Deterministically assign an ontology node type to each node index.

    Communities (if present) each map to one node type (round-robin), so a
    block/community becomes a typed sub-population; otherwise types are assigned
    round-robin over node indices.
    """
    node_types = ontology.node_types
    if not node_types:
        raise GraphGenerationError("ontology has no node types; cannot project topology")
    names = [nt.name for nt in node_types]
    if communities is not None:
        comm_ids = sorted(set(communities.values()))
        comm_to_type = {c: names[i % len(names)] for i, c in enumerate(comm_ids)}
        return [comm_to_type[communities[i]] for i in range(n)]
    return [names[i % len(names)] for i in range(n)]


def _edge_type_index(ontology: GraphOntology) -> dict[tuple[str, str], Any]:
    index: dict[tuple[str, str], Any] = {}
    for et in ontology.edge_types:
        index.setdefault((et.source_type, et.target_type), et)
        if not et.directed:
            index.setdefault((et.target_type, et.source_type), et)
    return index


def map_edges(
    graph: nx.Graph,
    node_ids: list[str],
    node_type_of: list[str],
    ontology: GraphOntology,
) -> tuple[list[Edge], int]:
    """Project topology edges onto ontology edge types (domain/range aware)."""
    index = _edge_type_index(ontology)
    edges: list[Edge] = []
    dropped = 0
    seen: set[tuple[str, str, str]] = set()
    for u, v in graph.edges():
        tu, tv = node_type_of[u], node_type_of[v]
        et = index.get((tu, tv))
        s, t = u, v
        if et is None:
            et = index.get((tv, tu))
            s, t = v, u
        if et is None:
            dropped += 1
            continue
        sig = (et.name, node_ids[s], node_ids[t])
        if sig in seen:
            continue
        seen.add(sig)
        edges.append(
            Edge(
                id=f"{et.name}:{len(edges)}",
                type=et.name,
                source=node_ids[s],
                target=node_ids[t],
                properties={},
            )
        )
    return edges, dropped


class ProceduralTopologyGenerator:
    """Generates a graph's structure from a seeded networkx topology model.

    No LLM: constructed without a provider. Follows the platform generator shape
    ``generate(spec, start_index, count, seed, shard_index=0) -> GraphDataset``.
    ``count`` is the node budget; the ontology comes from
    ``spec.directives["ontology"]`` (proposed by GA at create time).
    """

    def generate(
        self,
        spec: DatasetSpec,
        start_index: int,
        count: int,
        seed: int,
        shard_index: int = 0,
    ) -> GraphDataset:
        ontology = ontology_from_spec(dict(spec.directives))
        rng = np.random.default_rng([seed, shard_index])
        fake = Faker()
        # Per-instance seed (not global `Faker.seed`) so concurrent shards each
        # get an independent, deterministic Faker without racing on shared state.
        fake.seed_instance(seed * 1_000_003 + shard_index * 7919 + start_index)

        name = str(spec.directives.get("topology", "barabasi_albert"))
        n = max(1, count)
        graph, communities = build_topology(name, n, dict(spec.directives), rng)
        n = graph.number_of_nodes()

        type_of_index = assign_node_types(n, ontology, communities)
        ordered = sorted(graph.nodes())
        # relabel to a contiguous 0..n-1 index so node_ids/type lists align
        remap = {orig: i for i, orig in enumerate(ordered)}
        graph = nx.relabel_nodes(graph, remap, copy=True)

        type_by_type_name = {nt.name: nt for nt in ontology.node_types}
        node_ids: list[str] = []
        nodes: list[Node] = []
        for i in range(n):
            ntype = type_of_index[i]
            # Shard-global id (incorporate `start_index`) so multi-shard
            # assembly never dedups distinct nodes; identical to `{ntype}:{i}`
            # for the single-shard case (start_index == 0).
            node_id = f"{ntype}:{start_index + i}"
            node_ids.append(node_id)
            nodes.append(
                Node(
                    id=node_id,
                    type=ntype,
                    properties=synthesize_node_properties(type_by_type_name[ntype], rng, fake),
                )
            )

        edges, dropped = map_edges(graph, node_ids, type_of_index, ontology)
        metrics = compute_metrics(nodes, edges)
        metrics["engine"] = "topology"
        metrics["topology"] = name
        metrics["edges_dropped_unmappable"] = dropped
        if communities is not None:
            metrics["community_count"] = len(set(communities.values()))
        return GraphDataset(ontology=ontology, nodes=nodes, edges=edges, metrics=metrics)
