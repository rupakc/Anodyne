from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import SemanticType
from anodyne_generation.proposer import LLMSchemaProposer, SchemaProposalError


class _Provider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._c = content

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._c, usage=Usage())

    async def _stream_impl(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        # Not used in these tests
        return self._stream_impl()


_CFG = ModelConfig(
    id=uuid4(),
    tenant_id=uuid4(),
    name="m",
    provider="ollama",
    model="llama3",
)


@pytest.mark.asyncio
async def test_parses_valid_schema() -> None:
    content = '[{"name":"age","semantic_type":"integer","constraints":{"min":0,"max":120}}]'
    fields = await LLMSchemaProposer(_Provider(content), _CFG).propose("people with ages")
    assert fields[0].name == "age" and fields[0].semantic_type is SemanticType.INTEGER


@pytest.mark.asyncio
async def test_malformed_raises() -> None:
    with pytest.raises(SchemaProposalError):
        await LLMSchemaProposer(_Provider("not json"), _CFG).propose("x")


@pytest.mark.asyncio
async def test_extracts_json_from_fenced_block() -> None:
    content = 'Sure!\n```json\n[{"name":"n","semantic_type":"name"}]\n```'
    fields = await LLMSchemaProposer(_Provider(content), _CFG).propose("names")
    assert fields[0].semantic_type is SemanticType.NAME
