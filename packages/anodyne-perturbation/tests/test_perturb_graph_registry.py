"""Graph perturbation dispatch: the graph handler self-registers and the
`RegistryPerturbator` routes `Modality.GRAPH` to `anodyne_graph.perturb`.

The graph seam is deliberately separate from the columnar (pa.Table) registry:
a graph artifact is node-link JSON, so it must not be forced through a
`pyarrow.Table`. These tests are fully offline and deterministic.
"""

from __future__ import annotations

from anodyne_dataset.models import PerturbationFamily, PerturbationSpec
from anodyne_graph.models import Edge, EdgeType, GraphDataset, GraphOntology, Node, NodeType
from anodyne_graph.serialization import to_json_bytes
from anodyne_perturbation import RegistryPerturbator
from anodyne_perturbation.registry import (
    get_graph_perturbation_handler,
    registered_graph_perturbation_modalities,
)


def _dataset() -> GraphDataset:
    return GraphDataset(
        ontology=GraphOntology(
            node_types=[NodeType(name="Person"), NodeType(name="Company")],
            edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
        ),
        nodes=[
            Node(id="p0", type="Person"),
            Node(id="p1", type="Person"),
            Node(id="c0", type="Company"),
        ],
        edges=[
            Edge(id="e0", type="WORKS_AT", source="p0", target="c0"),
            Edge(id="e1", type="WORKS_AT", source="p1", target="c0"),
        ],
    )


def test_graph_modality_registered() -> None:
    assert "graph" in registered_graph_perturbation_modalities()


def test_graph_handler_perturbs_via_anodyne_graph() -> None:
    handler = get_graph_perturbation_handler("graph")
    spec = PerturbationSpec(family=PerturbationFamily.GRAPH_DROPOUT, intensity=1.0)
    out = handler.perturb(spec, _dataset(), 3)
    assert isinstance(out, GraphDataset)
    assert out.edges == []  # every edge dropped at intensity 1


def test_registry_perturbator_routes_graph() -> None:
    ds = _dataset()
    perturbator = RegistryPerturbator()
    spec = PerturbationSpec(family=PerturbationFamily.GRAPH_REWIRE, intensity=1.0)
    out = perturbator.perturb_graph(spec, ds, 3)
    assert isinstance(out, GraphDataset)
    assert {n.id for n in out.edges} != {n.id for n in ds.edges} or to_json_bytes(
        out
    ) != to_json_bytes(ds)
    assert {n.id for n in out.nodes} == {n.id for n in ds.nodes}


def test_graph_seam_absent_from_columnar_registry() -> None:
    from anodyne_perturbation import registered_perturbation_modalities

    # `graph` must NOT leak into the pa.Table registry (it has its own).
    assert "graph" not in registered_perturbation_modalities()
