from __future__ import annotations

import numpy as np
from anodyne_graph.constraints import (
    OntologyConstraintValidator,
    inject_violations,
)
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)

_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(
            name="Person",
            properties=[
                PropertySpec(name="handle", datatype="string"),
                PropertySpec(name="age", datatype="integer", constraints={"min": 0, "max": 120}),
                PropertySpec(name="tier", datatype="string", constraints={"choices": ["a", "b"]}),
            ],
        ),
        NodeType(name="Company", properties=[PropertySpec(name="handle", datatype="string")]),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)


def _conforming() -> GraphDataset:
    nodes = [
        Node(id="Person:0", type="Person", properties={"handle": "p0", "age": 30, "tier": "a"}),
        Node(id="Person:1", type="Person", properties={"handle": "p1", "age": 41, "tier": "b"}),
        Node(id="Company:0", type="Company", properties={"handle": "c0"}),
    ]
    edges = [
        Edge(id="WORKS_AT:0", type="WORKS_AT", source="Person:0", target="Company:0"),
        Edge(id="WORKS_AT:1", type="WORKS_AT", source="Person:1", target="Company:0"),
    ]
    return GraphDataset(ontology=_ONTOLOGY, nodes=nodes, edges=edges)


def test_conforming_graph_passes_check() -> None:
    report = OntologyConstraintValidator().check(_conforming())
    assert report.conforms
    assert report.count == 0


def test_domain_range_violation_detected() -> None:
    ds = _conforming()
    ds.edges.append(Edge(id="bad", type="WORKS_AT", source="Company:0", target="Person:0"))
    report = OntologyConstraintValidator().check(ds)
    assert not report.conforms
    assert any(v.kind == "domain_range" for v in report.violations)


def test_missing_required_and_choices_detected() -> None:
    ds = _conforming()
    ds.nodes[0].properties.pop("age")  # required missing
    ds.nodes[1].properties["tier"] = "z"  # not in choices
    report = OntologyConstraintValidator().check(ds)
    kinds = {v.kind for v in report.violations}
    assert "missing_required" in kinds
    assert "not_in_choices" in kinds


def test_cardinality_violation_detected() -> None:
    ds = _conforming()  # Person:1 already works at Company:0
    ds.edges.append(Edge(id="c2", type="WORKS_AT", source="Person:1", target="Company:0b"))
    ds.nodes.append(Node(id="Company:0b", type="Company", properties={"handle": "c0b"}))
    report = OntologyConstraintValidator().check(ds, {"WORKS_AT": {"max_out_per_source": 1}})
    assert any(v.kind == "cardinality" for v in report.violations)


def test_inject_violations_creates_n_dangling_edges() -> None:
    ds = _conforming()
    injected = inject_violations(ds, 3, np.random.default_rng([1, 0, 99]))
    assert injected.metrics["injected_violations"] == 3
    report = OntologyConstraintValidator().check(injected)
    assert report.count >= 3
    assert all(v.kind in ("dangling_edge", "domain_range") for v in report.violations)


def test_shacl_conforms_on_clean_fails_on_injected() -> None:
    validator = OntologyConstraintValidator()
    clean = validator.validate_shacl(_conforming())
    assert clean.conforms
    injected = inject_violations(_conforming(), 2, np.random.default_rng([2, 0, 99]))
    bad = validator.validate_shacl(injected)
    assert not bad.conforms
