from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_graph.graphrag.models import GraphQAItem
from anodyne_graph.graphrag.question_gen import (
    AGGREGATION_COUNT,
    CHAINED_RELATION,
    COMPARISON,
    EXISTENCE_NEGATION,
    FAMILIES,
    GraphRAGGenerator,
    _rel_phrase,
)
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
)


class _Provider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._c = content
        self.calls = 0

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self._c, usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")


def _dataset() -> GraphDataset:
    ontology = GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[]),
            NodeType(name="Company", properties=[]),
        ],
        edge_types=[
            EdgeType(name="KNOWS", source_type="Person", target_type="Person"),
            EdgeType(name="WORKS_AT", source_type="Person", target_type="Company"),
        ],
    )
    nodes = [
        Node(id=f"Person:{i}", type="Person", properties={"name": f"P{i}"}) for i in range(6)
    ] + [Node(id=f"Company:{i}", type="Company", properties={"name": f"C{i}"}) for i in range(2)]
    edges = [
        Edge(id="K0", type="KNOWS", source="Person:0", target="Person:1"),
        Edge(id="K1", type="KNOWS", source="Person:1", target="Person:2"),
        Edge(id="K2", type="KNOWS", source="Person:2", target="Person:3"),
        Edge(id="K3", type="KNOWS", source="Person:0", target="Person:4"),
        Edge(id="K4", type="KNOWS", source="Person:0", target="Person:5"),
        Edge(id="K5", type="KNOWS", source="Person:3", target="Person:4"),
        Edge(id="W0", type="WORKS_AT", source="Person:3", target="Company:0"),
        Edge(id="W1", type="WORKS_AT", source="Person:1", target="Company:1"),
        Edge(id="W2", type="WORKS_AT", source="Person:0", target="Company:0"),
    ]
    return GraphDataset(ontology=ontology, nodes=nodes, edges=edges)


def _label(ds: GraphDataset, node_id: str) -> str:
    node = next(n for n in ds.nodes if n.id == node_id)
    v = node.properties.get("name")
    return v if isinstance(v, str) else node.id


def _typed_neighbors(ds: GraphDataset, node_id: str, rel_type: str) -> list[str]:
    directed = {et.name: et.directed for et in ds.ontology.edge_types}
    out: list[str] = []
    for e in ds.edges:
        if e.type != rel_type:
            continue
        if e.source == node_id:
            out.append(e.target)
        elif not directed.get(e.type, True) and e.target == node_id:
            out.append(e.source)
    return out


def _degree(ds: GraphDataset, node_id: str) -> int:
    return sum((e.source == node_id) + (e.target == node_id) for e in ds.edges)


def _rel_type_from_question(ds: GraphDataset, question: str) -> str | None:
    """Recover the asked relation type from a *template* question surface.

    Matches each ontology edge type by its humanized phrase (the same
    `_rel_phrase` the generator uses). Returns ``None`` when the phrase is not
    present (e.g. an LLM-rewritten surface), so callers can fall back.
    """
    best: str | None = None
    for et in ds.ontology.edge_types:
        phrase = _rel_phrase(et.name)
        if phrase in question and (best is None or len(phrase) > len(_rel_phrase(best))):
            best = et.name
    return best


def _verify_answer_grounded(item: GraphQAItem, ds: GraphDataset) -> None:
    """Recompute the expected gold answer from the graph and assert it matches."""
    path = item.gold_path
    start = path.start_node_id
    terminal = path.terminal_node_id
    edges_by_id = {e.id: e for e in ds.edges}

    if item.question_type == CHAINED_RELATION:
        assert item.answer == _label(ds, terminal)
        assert item.answer_node_ids == [terminal]
    elif item.question_type == AGGREGATION_COUNT:
        rel = edges_by_id[path.edge_ids[0]].type
        nbrs = sorted(_typed_neighbors(ds, start, rel))
        assert item.answer == str(len(nbrs))
        assert item.answer_node_ids == nbrs
    elif item.question_type == EXISTENCE_NEGATION:
        assert item.answer in ("Yes", "No")
        rel_type = _rel_type_from_question(ds, item.question)
        if rel_type is not None:
            # Verify BOTH answers against the same undirected-aware traversal the
            # generator uses, so a "No" for a relation the node actually has on
            # the *target* side of an undirected edge can no longer pass silently.
            nbrs = sorted(_typed_neighbors(ds, start, rel_type))
            if item.answer == "Yes":
                assert nbrs, "a Yes existence answer must have real typed neighbors"
                assert item.answer_node_ids == nbrs
            else:
                assert nbrs == [], (
                    "a No existence answer must be truly absent under the "
                    "undirected-aware traversal"
                )
                assert item.answer_node_ids == []
        elif item.answer == "Yes":  # LLM-rewritten surface: structural fallback
            assert item.answer_node_ids  # non-empty, all real neighbors of start
            all_nbrs = {e.target for e in ds.edges if e.source == start} | {
                e.source for e in ds.edges if e.target == start
            }
            assert set(item.answer_node_ids) <= all_nbrs
        else:
            assert item.answer_node_ids == []
    elif item.question_type == COMPARISON:
        ds_deg, dt_deg = _degree(ds, start), _degree(ds, terminal)
        if ds_deg > dt_deg:
            assert item.answer == _label(ds, start)
        elif dt_deg > ds_deg:
            assert item.answer == _label(ds, terminal)
        else:
            assert "equal" in item.answer
    else:  # pragma: no cover - guards against a new unverified family
        raise AssertionError(f"unknown family {item.question_type}")


def _undirected_target_dataset() -> GraphDataset:
    """All-undirected FRIEND graph where several nodes appear only as edge
    *targets* (never a `source`). Under undirected semantics they still have the
    relation, so an existence question about them must answer "Yes"."""
    ontology = GraphOntology(
        node_types=[NodeType(name="Person", properties=[])],
        edge_types=[
            EdgeType(name="FRIEND", source_type="Person", target_type="Person", directed=False)
        ],
    )
    nodes = [Node(id=f"P{i}", type="Person", properties={"name": f"P{i}"}) for i in range(5)]
    edges = [
        Edge(id="F0", type="FRIEND", source="P0", target="P1"),
        Edge(id="F1", type="FRIEND", source="P0", target="P2"),
        Edge(id="F2", type="FRIEND", source="P1", target="P3"),
        Edge(id="F3", type="FRIEND", source="P2", target="P4"),
    ]
    return GraphDataset(ontology=ontology, nodes=nodes, edges=edges)


async def test_templates_cover_all_four_families() -> None:
    gen = GraphRAGGenerator()
    fixture = await gen.generate(_dataset(), "v1", num_questions=12, seed=3)
    assert set(FAMILIES) <= {item.question_type for item in fixture.items}


async def test_every_answer_matches_graph_ground_truth() -> None:
    ds = _dataset()
    fixture = await GraphRAGGenerator().generate(ds, "v1", num_questions=12, seed=3)
    for item in fixture.items:
        _verify_answer_grounded(item, ds)
        # gold path hops are real edges
        edges_by_id = {e.id: e for e in ds.edges}
        for eid in item.gold_path.edge_ids:
            assert eid in edges_by_id


async def test_existence_negation_grounded_for_undirected_target_side() -> None:
    """Finding 1: a node reached only on the *target* side of an undirected
    relation still HAS that relation. Every existence answer must be grounded in
    the same undirected-aware traversal -- a "No" is only valid when the relation
    is truly absent for the subject."""
    ds = _undirected_target_dataset()
    target_only = {"P3", "P4"}  # never a `source`, only reached as an edge target
    checked_target_only = False
    for seed in range(40):
        fixture = await GraphRAGGenerator().generate(ds, "v1", num_questions=8, seed=seed)
        for item in fixture.items:
            if item.question_type != EXISTENCE_NEGATION:
                continue
            # Grounds BOTH Yes and No against the undirected-aware traversal; the
            # old source-only logic produced a bogus "No" here for target-only
            # subjects, which this now catches.
            _verify_answer_grounded(item, ds)
            if item.gold_path.start_node_id in target_only:
                assert item.answer == "Yes"
                checked_target_only = True
    assert checked_target_only, "test never exercised a target-only existence subject"


async def test_fixture_is_deterministic_across_seeded_runs() -> None:
    ds = _dataset()
    a = await GraphRAGGenerator().generate(ds, "v1", num_questions=10, seed=99)
    b = await GraphRAGGenerator().generate(ds, "v1", num_questions=10, seed=99)
    assert a.model_dump() == b.model_dump()
    assert a.metadata["seed"] == 99


async def test_llm_rewrites_only_surface_answer_stays_grounded() -> None:
    ds = _dataset()
    bogus = "According to me the answer is TOTALLY_WRONG_42."
    provider = _Provider(bogus)
    gen = GraphRAGGenerator(provider, _CFG)
    fixture = await gen.generate(ds, "v1", num_questions=12, seed=5)
    assert provider.calls == len(fixture.items)
    for item in fixture.items:
        # the LLM output became the question surface, never the answer
        assert item.question == bogus
        assert item.answer != bogus
        _verify_answer_grounded(item, ds)
    assert fixture.metadata["llm_refined"] is True


async def test_empty_llm_output_falls_back_to_template_phrasing() -> None:
    ds = _dataset()
    templated = await GraphRAGGenerator().generate(ds, "v1", num_questions=8, seed=11)
    with_empty_llm = await GraphRAGGenerator(_Provider("   "), _CFG).generate(
        ds, "v1", num_questions=8, seed=11
    )
    # blank LLM output => identical question text to the no-provider template run
    assert [i.question for i in templated.items] == [i.question for i in with_empty_llm.items]
