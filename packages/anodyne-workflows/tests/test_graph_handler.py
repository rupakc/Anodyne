from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast
from uuid import UUID, uuid4

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider, ObjectStore
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_dataset.ports import DatasetRepository
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec
from anodyne_graph.serialization import from_json_bytes
from anodyne_workflows.activities import ActivityContext
from anodyne_workflows.handlers import GraphHandler
from anodyne_workflows.modality import get_handler
from anodyne_workflows.workflow import GenerationInput


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


class _Store(ObjectStore):
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.data[key] = data

    async def get(self, key: str) -> bytes:
        return self.data[key]

    async def presigned_url(self, key: str, expires: int = 86400) -> str:
        return key

    async def list(self, prefix: str) -> list[str]:
        return [k for k in self.data if k.startswith(prefix)]


class _Registry:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        return self._cfg if config_id == self._cfg.id else None


class _Repo:
    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        return None


_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(name="Person", properties=[PropertySpec(name="name", datatype="string")]),
        NodeType(name="Company", properties=[PropertySpec(name="name", datatype="string")]),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)

_GRAPH_JSON = json.dumps(
    {
        "nodes": [
            {"type": "Person", "key": "alice", "properties": {"name": "A"}},
            {"type": "Company", "key": "acme", "properties": {"name": "Acme"}},
        ],
        "edges": [{"type": "WORKS_AT", "source": "alice", "target": "acme", "properties": {}}],
    }
)


def _spec(tenant_id: UUID, dataset_id: UUID) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="g",
        description="people and companies",
        modality=Modality.GRAPH,
        source="description",
        fields=[],
        target_rows=10,
        directives={"ontology": _ONTOLOGY.model_dump(mode="json")},
    )


def _ctx(cfg: ModelConfig) -> ActivityContext:
    return ActivityContext(
        repo=cast(DatasetRepository, _Repo()),
        s3_bucket="b",
        s3_client=None,
        model_registry=_Registry(cfg),
        llm_provider=_Provider(_GRAPH_JSON),
    )


def _inp(tenant_id: UUID, dataset_id: UUID, cfg: ModelConfig) -> GenerationInput:
    return GenerationInput(
        job_id=str(uuid4()),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=10,
        seed=3,
        modality="graph",
        model_config_id=str(cfg.id),
    )


def test_graph_handler_is_registered() -> None:
    assert isinstance(get_handler("graph"), GraphHandler)
    assert get_handler("graph").artifact_format == "graph_json"


async def test_generate_and_assemble_produces_graph_artifact() -> None:
    tenant_id, dataset_id = uuid4(), uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tenant_id, name="m", provider="gemini", model="g")
    handler = GraphHandler()
    ctx = _ctx(cfg)
    inp = _inp(tenant_id, dataset_id, cfg)
    spec = _spec(tenant_id, dataset_id)
    store = _Store()

    keys = await handler.generate_shards(ctx, inp, spec, [[0, 10]], store)
    assert len(keys) == 1

    artifact_key = await handler.assemble(ctx, inp, spec, keys, store)
    dataset = from_json_bytes(store.data[artifact_key])
    assert {n.id for n in dataset.nodes} == {"Person:alice", "Company:acme"}
    assert len(dataset.edges) == 1
    assert dataset.metrics["node_count"] == 2
    assert dataset.ontology.edge_type("WORKS_AT") is not None


async def test_assemble_dedups_nodes_across_shards() -> None:
    tenant_id, dataset_id = uuid4(), uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tenant_id, name="m", provider="gemini", model="g")
    handler = GraphHandler()
    ctx = _ctx(cfg)
    inp = _inp(tenant_id, dataset_id, cfg)
    spec = _spec(tenant_id, dataset_id)
    store = _Store()

    # Two shards with identical content -> merged graph must not double nodes/edges.
    keys = await handler.generate_shards(ctx, inp, spec, [[0, 10], [10, 10]], store)
    assert len(keys) == 2
    artifact_key = await handler.assemble(ctx, inp, spec, keys, store)
    dataset = from_json_bytes(store.data[artifact_key])
    assert dataset.metrics["node_count"] == 2
    assert dataset.metrics["edge_count"] == 1
