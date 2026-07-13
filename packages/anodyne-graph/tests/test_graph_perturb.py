"""Graph perturbations (Track GH): structural + semantic families.

Everything is offline, seeded and deterministic; each test asserts one contract
from the plan's GH section (no-op at intensity 0, expected affected fraction at
intensity 1, node/degree preservation, dangling-edge cleanup, injected ontology
violations, determinism, and non-mutation of the input).
"""

from __future__ import annotations

from anodyne_dataset.models import PerturbationFamily
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
from anodyne_graph.perturb import perturb_graph
from anodyne_graph.serialization import to_json_bytes

GRAPH_FAMILIES = [
    PerturbationFamily.GRAPH_REWIRE,
    PerturbationFamily.GRAPH_DROPOUT,
    PerturbationFamily.GRAPH_ONTOLOGY_VIOLATION,
]


def _ontology() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(
                name="Person",
                properties=[
                    PropertySpec(
                        name="age", datatype="integer", constraints={"min": 0, "max": 120}
                    ),
                    PropertySpec(
                        name="status",
                        datatype="string",
                        constraints={"choices": ["active", "inactive"]},
                    ),
                    PropertySpec(name="full_name", datatype="string"),
                ],
            ),
            NodeType(name="Company"),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )


def _dataset() -> GraphDataset:
    persons = [
        Node(
            id=f"p{i}",
            type="Person",
            properties={"age": 20 + i, "status": "active", "full_name": f"Faked Name {i}"},
        )
        for i in range(6)
    ]
    companies = [Node(id=f"c{j}", type="Company", properties={}) for j in range(3)]
    edges = [
        Edge(id=f"e{i}", type="WORKS_AT", source=f"p{i}", target=f"c{i % 3}") for i in range(6)
    ]
    nodes = persons + companies
    return GraphDataset(
        ontology=_ontology(), nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def _node_ids(ds: GraphDataset) -> set[str]:
    return {n.id for n in ds.nodes}


def _degree_seq(ds: GraphDataset) -> tuple[dict[str, int], dict[str, int]]:
    out_deg: dict[str, int] = {n.id: 0 for n in ds.nodes}
    in_deg: dict[str, int] = {n.id: 0 for n in ds.nodes}
    for e in ds.edges:
        out_deg[e.source] = out_deg.get(e.source, 0) + 1
        in_deg[e.target] = in_deg.get(e.target, 0) + 1
    return out_deg, in_deg


def _has_dangling(ds: GraphDataset) -> bool:
    ids = _node_ids(ds)
    return any(e.source not in ids or e.target not in ids for e in ds.edges)


def test_intensity_zero_is_noop_for_every_family() -> None:
    ds = _dataset()
    before = to_json_bytes(ds)
    for family in GRAPH_FAMILIES:
        out = perturb_graph(ds, family, intensity=0.0, seed=3, params={})
        assert to_json_bytes(out) == before


def test_input_is_never_mutated() -> None:
    ds = _dataset()
    before = to_json_bytes(ds)
    for family in GRAPH_FAMILIES:
        perturb_graph(ds, family, intensity=1.0, seed=5, params={})
    assert to_json_bytes(ds) == before


def test_determinism_same_seed_byte_identical() -> None:
    ds = _dataset()
    for family in GRAPH_FAMILIES:
        a = perturb_graph(ds, family, intensity=0.5, seed=11, params={})
        b = perturb_graph(ds, family, intensity=0.5, seed=11, params={})
        assert to_json_bytes(a) == to_json_bytes(b)


def test_rewire_preserves_node_set_and_edge_count() -> None:
    ds = _dataset()
    out = perturb_graph(ds, PerturbationFamily.GRAPH_REWIRE, intensity=1.0, seed=2, params={})
    assert _node_ids(out) == _node_ids(ds)
    assert len(out.edges) == len(ds.edges)
    # Something actually moved.
    assert {(e.source, e.target) for e in out.edges} != {(e.source, e.target) for e in ds.edges}


def test_rewire_degree_preserving_keeps_degree_sequence() -> None:
    ds = _dataset()
    out = perturb_graph(
        ds,
        PerturbationFamily.GRAPH_REWIRE,
        intensity=1.0,
        seed=2,
        params={"degree_preserving": True},
    )
    assert _degree_seq(out) == _degree_seq(ds)


def test_dropout_removes_expected_edge_fraction_no_dangling() -> None:
    ds = _dataset()
    out = perturb_graph(
        ds, PerturbationFamily.GRAPH_DROPOUT, intensity=0.5, seed=4, params={"target": "edges"}
    )
    assert len(out.edges) == len(ds.edges) - round(0.5 * len(ds.edges))
    assert _node_ids(out) == _node_ids(ds)
    assert not _has_dangling(out)


def test_dropout_nodes_removes_dangling_edges() -> None:
    ds = _dataset()
    out = perturb_graph(
        ds, PerturbationFamily.GRAPH_DROPOUT, intensity=0.5, seed=4, params={"target": "nodes"}
    )
    assert len(out.nodes) == len(ds.nodes) - round(0.5 * len(ds.nodes))
    assert not _has_dangling(out)


def test_dropout_intensity_one_removes_all_edges() -> None:
    ds = _dataset()
    out = perturb_graph(
        ds, PerturbationFamily.GRAPH_DROPOUT, intensity=1.0, seed=4, params={"target": "edges"}
    )
    assert out.edges == []


def test_ontology_violation_injects_detectable_violations() -> None:
    ds = _dataset()
    out = perturb_graph(
        ds, PerturbationFamily.GRAPH_ONTOLOGY_VIOLATION, intensity=1.0, seed=6, params={}
    )
    onto = out.ontology
    by_id = {n.id: n for n in out.nodes}
    node_bad = sum(1 for n in out.nodes if onto.node_type(n.type) is None)
    edge_bad = 0
    for e in out.edges:
        et = onto.edge_type(e.type)
        src, tgt = by_id.get(e.source), by_id.get(e.target)
        if not (
            et is not None
            and src is not None
            and tgt is not None
            and src.type == et.source_type
            and tgt.type == et.target_type
        ):
            edge_bad += 1
    # Property-level violations (bad value against declared constraints) also count.
    prop_bad = 0
    for n in out.nodes:
        nt = onto.node_type(n.type)
        if nt is None:
            continue
        for prop in nt.properties:
            v = n.properties.get(prop.name)
            if prop.name == "age" and isinstance(v, int) and (v < 0 or v > 120):
                prop_bad += 1
            if prop.name == "status" and v is not None and v not in ("active", "inactive"):
                prop_bad += 1
    assert node_bad + edge_bad + prop_bad > 0


def test_ontology_violation_keeps_pii_faked() -> None:
    ds = _dataset()
    original = {n.id: n.properties.get("full_name") for n in ds.nodes if n.type == "Person"}
    out = perturb_graph(
        ds, PerturbationFamily.GRAPH_ONTOLOGY_VIOLATION, intensity=1.0, seed=6, params={}
    )
    # PII values are never newly synthesized/corrupted: the faked name stays as
    # generated for any node still carrying it.
    for n in out.nodes:
        if n.type == "Person" and "full_name" in n.properties:
            assert n.properties["full_name"] == original[n.id]
