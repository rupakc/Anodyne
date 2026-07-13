"""Shared graph-judge test fixtures (sub-system GD).

Exposed as pytest fixtures (auto-discovered regardless of ``--import-mode``) so
the individually-named ``test_graph_judge_*.py`` files stay dependency-free and
have no ``tests/__init__.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import uuid4

import networkx as nx  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import Modality
from anodyne_evaluation.ports import EvaluationContext
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)

NodeTypeFn = Callable[[object, int], str]


def _dataset_from_nx(
    g: nx.Graph,
    *,
    node_type_fn: NodeTypeFn | None = None,
    edge_type: str = "LINK",
    extra_edge_types: Iterable[str] = (),
    extra_node_types: Iterable[str] = (),
) -> GraphDataset:
    """Build a `GraphDataset` from a networkx graph with a trivial ontology."""
    type_of: dict[str, str] = {}
    nodes: list[Node] = []
    for x in g.nodes():
        ntype = node_type_fn(x, g.degree(x)) if node_type_fn else "Item"
        type_of[str(x)] = ntype
        nodes.append(Node(id=str(x), type=ntype))
    edges = [
        Edge(id=f"e{i}", type=edge_type, source=str(u), target=str(v))
        for i, (u, v) in enumerate(g.edges())
    ]
    node_type_names = sorted({*type_of.values(), *extra_node_types})
    edge_source, edge_target = (
        (node_type_names[0], node_type_names[0]) if node_type_names else ("Item", "Item")
    )
    ontology = GraphOntology(
        node_types=[NodeType(name=n) for n in node_type_names],
        edge_types=[
            EdgeType(name=edge_type, source_type=edge_source, target_type=edge_target),
            *(
                EdgeType(name=t, source_type=edge_source, target_type=edge_target)
                for t in extra_edge_types
            ),
        ],
    )
    return GraphDataset(ontology=ontology, nodes=nodes, edges=edges)


@pytest.fixture
def dataset_from_nx() -> Callable[..., GraphDataset]:
    return _dataset_from_nx


@pytest.fixture
def graph_context() -> Callable[..., EvaluationContext]:
    def _make(
        subject_graph: GraphDataset,
        reference_graph: GraphDataset | None = None,
        *,
        seed: int = 0,
        sample_rows: int = 20,
        metadata: dict[str, str] | None = None,
    ) -> EvaluationContext:
        return EvaluationContext(
            subject=pd.DataFrame(),
            modality=Modality.GRAPH,
            subject_graph=subject_graph,
            reference_graph=reference_graph,
            seed=seed,
            sample_rows=sample_rows,
            metadata=metadata or {},
        )

    return _make


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="openai", model="gpt-4o")


# Re-export the model classes so tests can build bespoke ontologies/graphs.
__all__ = [
    "Edge",
    "EdgeType",
    "GraphDataset",
    "GraphOntology",
    "Node",
    "NodeType",
    "PropertySpec",
]
