"""Structural-fidelity expert: synthetic graph vs a reference graph.

Six complementary structural signals, each turned into a 0..1 *similarity* and
averaged (higher == closer to the reference):

- **degree distribution** — two-sample KS statistic on the degree sequences;
- **average clustering coefficient** — absolute delta;
- **degree assortativity** — absolute delta (range [-1, 1], halved);
- **modularity** — greedy-community modularity, absolute delta;
- **path-length** — average shortest-path length on the giant component,
  relative delta;
- **spectral distance** — the chosen *scalable* whole-graph distance: L2 distance
  between the sorted **normalized-Laplacian eigenvalue** spectra (padded to equal
  length, normalized by the eigenvalue range [0, 2] and spectrum size).

Graph edit distance / isomorphism is deliberately avoided (NP-hard, does not
scale — see spec §6). The eigendecomposition is O(n^3), so for large graphs the
spectrum is computed on a deterministic node-induced subgraph (`_SPECTRAL_MAX`);
the honest scaling lever is a sparse partial-spectrum solver, noted for later.
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

from anodyne_evaluation.graph_judges.base import GraphJudge, clamp01, require_graph, to_undirected
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable

_SPECTRAL_MAX = 400


def _assortativity(g: nx.Graph) -> float:
    if g.number_of_edges() == 0:
        return 0.0
    try:
        val = float(nx.degree_assortativity_coefficient(g))
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return 0.0
    return 0.0 if not np.isfinite(val) else val


def _modularity(g: nx.Graph) -> float:
    if g.number_of_edges() == 0:
        return 0.0
    try:
        communities = nx.community.greedy_modularity_communities(g)
        return float(nx.community.modularity(g, communities))
    except (ValueError, ZeroDivisionError, KeyError):
        return 0.0


def _avg_path_length(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2:
        return 0.0
    giant_nodes = max(nx.connected_components(g), key=len)
    giant = g.subgraph(giant_nodes)
    if giant.number_of_nodes() < 2:
        return 0.0
    return float(nx.average_shortest_path_length(giant))


def _spectrum(g: nx.Graph) -> np.ndarray:
    if g.number_of_nodes() == 0:
        return np.zeros(0)
    if g.number_of_nodes() > _SPECTRAL_MAX:
        g = g.subgraph(sorted(g.nodes())[:_SPECTRAL_MAX])
    return np.sort(np.asarray(nx.normalized_laplacian_spectrum(g), dtype=float))


def _resample(spectrum: np.ndarray, length: int) -> np.ndarray:
    # Linearly resample a sorted eigenvalue spectrum to `length` points. This
    # replaces zero-padding the shorter spectrum: padding let a node-count gap
    # dominate the L2 delta (two structurally-similar graphs of different sizes
    # scored as very distant). Resampling compares spectral *shape* at a common
    # resolution, so size differences no longer inflate the distance.
    if len(spectrum) == length:
        return spectrum
    if len(spectrum) == 0:
        return np.zeros(length)
    if len(spectrum) == 1:
        return np.full(length, spectrum[0])
    xp = np.linspace(0.0, 1.0, len(spectrum))
    x = np.linspace(0.0, 1.0, length)
    return np.asarray(np.interp(x, xp, spectrum), dtype=float)


def _spectral_distance(a: nx.Graph, b: nx.Graph) -> float:
    ea, eb = _spectrum(a), _spectrum(b)
    n = max(len(ea), len(eb))
    if n == 0:
        return 0.0
    ea = _resample(ea, n)
    eb = _resample(eb, n)
    # normalized-Laplacian eigenvalues live in [0, 2]; normalize the L2 delta by
    # sqrt(n) (per-eigenvalue RMS) and by 2 (the range) into [0, 1].
    return clamp01(float(np.linalg.norm(ea - eb)) / (np.sqrt(n) * 2.0))


class StructuralFidelityGraphJudge(GraphJudge):
    dimension = EvalDimension.GRAPH_STRUCTURE

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        syn = require_graph(ctx)
        if ctx.reference_graph is None:
            raise JudgeNotApplicable("structural fidelity requires a reference graph")
        gs = to_undirected(syn)
        gr = to_undirected(ctx.reference_graph)
        if gs.number_of_nodes() == 0 or gr.number_of_nodes() == 0:
            raise JudgeNotApplicable("structural fidelity requires non-empty graphs")

        deg_s = [d for _, d in gs.degree()]
        deg_r = [d for _, d in gr.degree()]
        deg_ks = float(ks_2samp(deg_s, deg_r).statistic) if deg_s and deg_r else 1.0

        clus_s, clus_r = float(nx.average_clustering(gs)), float(nx.average_clustering(gr))
        assort_s, assort_r = _assortativity(gs), _assortativity(gr)
        mod_s, mod_r = _modularity(gs), _modularity(gr)
        path_s, path_r = _avg_path_length(gs), _avg_path_length(gr)
        spectral = _spectral_distance(gs, gr)

        path_denom = max(path_s, path_r, 1.0)
        sims = {
            "degree_similarity": clamp01(1.0 - deg_ks),
            "clustering_similarity": clamp01(1.0 - abs(clus_s - clus_r)),
            "assortativity_similarity": clamp01(1.0 - abs(assort_s - assort_r) / 2.0),
            "modularity_similarity": clamp01(1.0 - abs(mod_s - mod_r)),
            "path_length_similarity": clamp01(1.0 - abs(path_s - path_r) / path_denom),
            "spectral_similarity": clamp01(1.0 - spectral),
        }
        score = clamp01(float(np.mean(list(sims.values()))))

        recs: list[str] = []
        if sims["degree_similarity"] < 0.8:
            recs.append(
                "Degree distribution drifts from the reference; revisit the topology model."
            )
        if sims["spectral_similarity"] < 0.8:
            recs.append("Global structure (Laplacian spectrum) diverges from the reference graph.")
        if sims["modularity_similarity"] < 0.8:
            recs.append("Community structure differs; tune the block/community generator.")

        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Structural fidelity vs reference: degree-KS={deg_ks:.3f}, "
                f"clustering {clus_s:.3f} vs {clus_r:.3f}, assortativity {assort_s:.3f} vs "
                f"{assort_r:.3f}, modularity {mod_s:.3f} vs {mod_r:.3f}, avg-path {path_s:.2f} "
                f"vs {path_r:.2f}, spectral-distance={spectral:.3f}."
            ),
            metrics={
                "degree_ks": deg_ks,
                "clustering_synthetic": clus_s,
                "clustering_reference": clus_r,
                "assortativity_synthetic": assort_s,
                "assortativity_reference": assort_r,
                "modularity_synthetic": mod_s,
                "modularity_reference": mod_r,
                "avg_path_length_synthetic": path_s,
                "avg_path_length_reference": path_r,
                "spectral_distance": spectral,
                **sims,
            },
            recommendations=recs,
        )
