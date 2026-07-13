"""Ontology-constrained validation: domain/range + cardinality + SHACL.

Two complementary checkers:

1. ``OntologyConstraintValidator.check`` -- a fast, pure-Python pass covering
   the constraints that live natively in the LPG ontology: edge **domain/range**
   (endpoints must exist and match the edge type's declared source/target),
   node **property** rules (required = non-nullable present & non-null; declared
   ``choices``/``min``/``max``), and optional **cardinality** (max out-degree per
   source node for a relation, supplied via ``directives["cardinality"]``).

2. ``OntologyConstraintValidator.validate_shacl`` -- projects the graph +
   ontology to RDF (``rdflib``) and validates with **SHACL** (``pyshacl``). The
   ontology becomes SHACL NodeShapes: one per node type, with property shapes
   for required datatype properties (``sh:minCount``/``sh:datatype``) and for
   each relation an ``sh:property`` with ``sh:class`` enforcing the range. This
   is the standards-based cross-check of (1); we run both because SHACL also
   validates the RDF projection that wave GC will export.

``inject_violations`` deliberately introduces N constraint violations (dangling
edges referencing non-existent nodes -- caught by both checkers) for robustness
testing; the count is recorded in ``metrics["injected_violations"]``.

No OWL reasoning / DL consistency here (that is a GD evaluation concern); this
module only enforces the ontology's structural constraints.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import numpy as np
from pydantic import BaseModel, Field

from anodyne_graph.models import Edge, GraphDataset, GraphOntology


class Violation(BaseModel):
    """One constraint breach: a machine ``kind`` + a human ``detail``."""

    kind: str
    detail: str
    subject: str = ""


class ConstraintReport(BaseModel):
    conforms: bool
    violations: list[Violation] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.violations)


class ShaclReport(BaseModel):
    conforms: bool
    count: int
    text: str = ""


_XSD = {
    "string": "http://www.w3.org/2001/XMLSchema#string",
    "integer": "http://www.w3.org/2001/XMLSchema#integer",
    "float": "http://www.w3.org/2001/XMLSchema#double",
    "boolean": "http://www.w3.org/2001/XMLSchema#boolean",
    "datetime": "http://www.w3.org/2001/XMLSchema#dateTime",
}


class OntologyConstraintValidator:
    """Validates a `GraphDataset` against its ontology (structural constraints)."""

    def check(
        self, dataset: GraphDataset, cardinality: dict[str, Any] | None = None
    ) -> ConstraintReport:
        ont = dataset.ontology
        node_type_of = {n.id: n.type for n in dataset.nodes}
        violations: list[Violation] = []

        self._check_edges(dataset, node_type_of, violations)
        self._check_properties(dataset, violations)
        if cardinality:
            self._check_cardinality(dataset, cardinality, violations)
        _ = ont
        return ConstraintReport(conforms=not violations, violations=violations)

    @staticmethod
    def _check_edges(
        dataset: GraphDataset, node_type_of: dict[str, str], violations: list[Violation]
    ) -> None:
        ont = dataset.ontology
        for e in dataset.edges:
            et = ont.edge_type(e.type)
            if et is None:
                violations.append(
                    Violation(
                        kind="unknown_edge_type", detail=f"edge type {e.type!r}", subject=e.id
                    )
                )
                continue
            if e.source not in node_type_of or e.target not in node_type_of:
                violations.append(
                    Violation(kind="dangling_edge", detail="missing endpoint", subject=e.id)
                )
                continue
            if node_type_of[e.source] != et.source_type or node_type_of[e.target] != et.target_type:
                violations.append(
                    Violation(
                        kind="domain_range",
                        detail=(
                            f"{et.name} expects {et.source_type}->{et.target_type}, got "
                            f"{node_type_of[e.source]}->{node_type_of[e.target]}"
                        ),
                        subject=e.id,
                    )
                )

    @staticmethod
    def _check_properties(dataset: GraphDataset, violations: list[Violation]) -> None:
        ont = dataset.ontology
        for n in dataset.nodes:
            nt = ont.node_type(n.type)
            if nt is None:
                violations.append(
                    Violation(kind="unknown_node_type", detail=f"type {n.type!r}", subject=n.id)
                )
                continue
            for prop in nt.properties:
                val = n.properties.get(prop.name)
                if val is None:
                    if not prop.nullable:
                        violations.append(
                            Violation(
                                kind="missing_required",
                                detail=f"property {prop.name!r}",
                                subject=n.id,
                            )
                        )
                    continue
                choices = prop.constraints.get("choices")
                if isinstance(choices, list) and choices and val not in choices:
                    violations.append(
                        Violation(
                            kind="not_in_choices", detail=f"{prop.name}={val!r}", subject=n.id
                        )
                    )
                if prop.datatype in ("integer", "float"):
                    _check_numeric_range(n.id, prop, val, violations)

    @staticmethod
    def _check_cardinality(
        dataset: GraphDataset, cardinality: dict[str, Any], violations: list[Violation]
    ) -> None:
        for etype, rule in cardinality.items():
            if not isinstance(rule, dict):
                continue
            max_out = rule.get("max_out_per_source")
            if max_out is None:
                continue
            counts: dict[str, int] = {}
            for e in dataset.edges:
                if e.type == etype:
                    counts[e.source] = counts.get(e.source, 0) + 1
            for src, c in counts.items():
                if c > int(max_out):
                    violations.append(
                        Violation(
                            kind="cardinality",
                            detail=f"{etype}: {c} > max {max_out}",
                            subject=src,
                        )
                    )

    # -- SHACL --------------------------------------------------------------
    def validate_shacl(self, dataset: GraphDataset) -> ShaclReport:
        from pyshacl import validate  # local import: heavy, optional path

        data_graph = self._to_rdf(dataset)
        shapes_graph = self._to_shapes(dataset.ontology)
        conforms, _results_graph, text = validate(
            data_graph,
            shacl_graph=shapes_graph,
            inference="none",
            advanced=True,
        )
        count = text.count("Constraint Violation") if not conforms else 0
        return ShaclReport(conforms=bool(conforms), count=count, text=str(text))

    @staticmethod
    def _uri(base: str, local: str) -> Any:
        from rdflib import URIRef

        return URIRef(base + quote(local, safe=""))

    def _to_rdf(self, dataset: GraphDataset) -> Any:
        from rdflib import Graph, Literal
        from rdflib.namespace import RDF

        ex = "urn:anodyne:graph#"
        g = Graph()
        for n in dataset.nodes:
            subj = self._uri(ex, n.id)
            g.add((subj, RDF.type, self._uri(ex, "class:" + n.type)))
            nt = dataset.ontology.node_type(n.type)
            dtypes = {p.name: p.datatype for p in (nt.properties if nt else [])}
            for k, v in n.properties.items():
                if v is None:
                    continue
                g.add((subj, self._uri(ex, "prop:" + k), Literal(v)))
                _ = dtypes
        for e in dataset.edges:
            g.add(
                (self._uri(ex, e.source), self._uri(ex, "rel:" + e.type), self._uri(ex, e.target))
            )
        return g

    def _to_shapes(self, ontology: GraphOntology) -> Any:
        from rdflib import BNode, Graph, Literal
        from rdflib.namespace import RDF, SH, XSD

        ex = "urn:anodyne:graph#"
        g = Graph()
        g.bind("sh", SH)
        for nt in ontology.node_types:
            shape = self._uri(ex, "shape:" + nt.name)
            g.add((shape, RDF.type, SH.NodeShape))
            g.add((shape, SH.targetClass, self._uri(ex, "class:" + nt.name)))
            for prop in nt.properties:
                pnode = BNode()
                g.add((shape, SH.property, pnode))
                g.add((pnode, SH.path, self._uri(ex, "prop:" + prop.name)))
                if not prop.nullable:
                    g.add((pnode, SH.minCount, Literal(1, datatype=XSD.integer)))
                dt = _XSD.get(prop.datatype)
                if dt is not None:
                    from rdflib import URIRef

                    g.add((pnode, SH.datatype, URIRef(dt)))
        # relation shapes: enforce range via sh:class on the source node shape
        for et in ontology.edge_types:
            shape = self._uri(ex, "shape:" + et.source_type)
            pnode = BNode()
            g.add((shape, SH.property, pnode))
            g.add((pnode, SH.path, self._uri(ex, "rel:" + et.name)))
            g.add((pnode, SH["class"], self._uri(ex, "class:" + et.target_type)))
        return g


def _check_numeric_range(node_id: str, prop: Any, val: Any, violations: list[Violation]) -> None:
    try:
        num = float(val)
    except (TypeError, ValueError):
        violations.append(
            Violation(kind="wrong_type", detail=f"{prop.name}={val!r} not numeric", subject=node_id)
        )
        return
    lo = prop.constraints.get("min")
    hi = prop.constraints.get("max")
    if lo is not None and num < float(lo):
        violations.append(
            Violation(kind="below_min", detail=f"{prop.name}={val} < {lo}", subject=node_id)
        )
    if hi is not None and num > float(hi):
        violations.append(
            Violation(kind="above_max", detail=f"{prop.name}={val} > {hi}", subject=node_id)
        )


def inject_violations(dataset: GraphDataset, n: int, rng: np.random.Generator) -> GraphDataset:
    """Return a copy of ``dataset`` with ``n`` deliberate domain/range violations.

    Each injected edge references a non-existent target node id, so it fails both
    the pure-Python domain/range check and the SHACL range shape. The count is
    recorded in ``metrics["injected_violations"]``.
    """
    if n <= 0 or not dataset.edges:
        return dataset
    extra: list[Edge] = []
    for i in range(n):
        template = dataset.edges[int(rng.integers(0, len(dataset.edges)))]
        extra.append(
            Edge(
                id=f"__violation__:{i}",
                type=template.type,
                source=template.source,
                target=f"__missing__:{i}",
                properties={},
            )
        )
    metrics = dict(dataset.metrics)
    metrics["injected_violations"] = n
    return dataset.model_copy(update={"edges": dataset.edges + extra, "metrics": metrics})
