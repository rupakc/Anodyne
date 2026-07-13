from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.errors import OntologyProposalError
from anodyne_graph.ontology import LLMOntologyProposer


class _Provider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._c = content

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._c, usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")


def _spec(description: str = "a social network of people and companies") -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description=description,
        modality=Modality.GRAPH,
        source="description",
        fields=[],
        target_rows=10,
    )


_VALID = (
    '{"node_types":[{"name":"Person","properties":[{"name":"age","datatype":"integer"}]},'
    '{"name":"Company"}],'
    '"edge_types":[{"name":"WORKS_AT","source_type":"Person","target_type":"Company"}]}'
)


async def test_parses_valid_ontology() -> None:
    onto = await LLMOntologyProposer().propose(_spec(), _Provider(_VALID), _CFG)
    assert [nt.name for nt in onto.node_types] == ["Person", "Company"]
    person = onto.node_type("Person")
    assert person is not None
    assert person.properties[0].datatype == "integer"
    assert onto.edge_types[0].source_type == "Person"


async def test_extracts_fenced_json() -> None:
    content = f"Sure!\n```json\n{_VALID}\n```"
    onto = await LLMOntologyProposer().propose(_spec(), _Provider(content), _CFG)
    assert onto.node_type("Company") is not None


async def test_drops_edges_with_unknown_endpoints() -> None:
    content = (
        '{"node_types":[{"name":"Person"}],'
        '"edge_types":[{"name":"BAD","source_type":"Person","target_type":"Ghost"},'
        '{"name":"KNOWS","source_type":"Person","target_type":"Person"}]}'
    )
    onto = await LLMOntologyProposer().propose(_spec(), _Provider(content), _CFG)
    assert [et.name for et in onto.edge_types] == ["KNOWS"]


async def test_raises_on_no_node_types() -> None:
    with pytest.raises(OntologyProposalError):
        await LLMOntologyProposer().propose(_spec(), _Provider('{"node_types":[]}'), _CFG)


async def test_raises_on_unparseable() -> None:
    with pytest.raises(OntologyProposalError):
        await LLMOntologyProposer().propose(_spec(), _Provider("not json"), _CFG)
