from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    Role,
    TenantContext,
    Usage,
    User,
)
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, FieldSpec, GenerationJob
from anodyne_dataset.ports import SchemaProposer
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec
from anodyne_workflows.workflow import GenerationInput
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")]),
        NodeType(name="Company"),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)


class _FakeOntologyProposer:
    async def propose(self, spec: DatasetSpec, provider: Any, config: Any) -> GraphOntology:
        return _ONTOLOGY


class _InertSchemaProposer(SchemaProposer):
    # Declared dependency of POST /datasets; never used on the graph branch.
    async def propose(self, description: str) -> list[FieldSpec]:
        return []


class _NullProvider(LLMProvider):
    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="", usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


def _handle() -> deps.OntologyProposalHandle:
    cfg = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")
    return deps.OntologyProposalHandle(_FakeOntologyProposer(), _NullProvider(), cfg)


class _FakeRepo:
    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.jobs: dict[UUID, GenerationJob] = {}

    async def create_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        spec = self.specs.get(dataset_id)
        return spec if spec is not None and spec.tenant_id == tenant_id else None

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job


class _FakeModelRegistry:
    def __init__(self) -> None:
        self._configs: list[ModelConfig] = []

    def add(self, cfg: ModelConfig) -> None:
        self._configs.append(cfg)

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        return next((c for c in self._configs if c.id == config_id), None)

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        return list(self._configs)


class _FakeHandle:
    def __init__(self, wf_id: str) -> None:
        self.id = wf_id


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(self, workflow: Any, arg: Any, **kwargs: Any) -> _FakeHandle:
        self.calls.append({"workflow": workflow, "arg": arg, **kwargs})
        return _FakeHandle(kwargs["id"])


class _FakeReviewRepo:
    async def create(self, task: Any) -> None:  # pragma: no cover - unused here
        pass


@pytest.fixture
def wired() -> tuple[AsyncClient, Any, _FakeRepo, _FakeTemporalClient, _FakeModelRegistry]:
    app = create_app()
    repo = _FakeRepo()
    fake_client = _FakeTemporalClient()
    registry = _FakeModelRegistry()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_schema_proposer] = lambda: _InertSchemaProposer()
    app.dependency_overrides[deps.get_ontology_proposer] = _handle
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    app.dependency_overrides[deps.get_model_registry] = lambda: registry
    app.dependency_overrides[deps.get_review_repo] = lambda: _FakeReviewRepo()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo, fake_client, registry


async def test_create_graph_dataset_returns_proposed_ontology(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        "/datasets",
        json={
            "name": "kg",
            "description": "people and companies",
            "target_rows": 20,
            "modality": "graph",
        },
    )

    assert r.status_code == 201
    body = r.json()
    assert body["modality"] == "graph"
    ontology = body["directives"]["ontology"]
    assert [nt["name"] for nt in ontology["node_types"]] == ["Person", "Company"]
    assert ontology["edge_types"][0]["source_type"] == "Person"
    # spec persisted with the ontology in directives (no DB table).
    stored = repo.specs[UUID(body["id"])]
    assert stored.directives["ontology"]["node_types"][0]["name"] == "Person"


async def test_generate_graph_defaults_to_gemini_model(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, fake_client, registry = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    ollama = ModelConfig(id=uuid4(), tenant_id=tid, name="o", provider="ollama", model="l")
    gemini = ModelConfig(id=uuid4(), tenant_id=tid, name="g", provider="gemini", model="g")
    registry.add(ollama)
    registry.add(gemini)

    created = await client.post(
        "/datasets",
        json={"name": "kg", "description": "x", "target_rows": 20, "modality": "graph"},
    )
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 5})
    assert r.status_code == 202
    inp = fake_client.calls[0]["arg"]
    assert isinstance(inp, GenerationInput)
    assert inp.modality == "graph"
    # default-model pick applies to graph too: the Gemini config, not the first (ollama).
    assert inp.model_config_id == str(gemini.id)


async def test_generate_graph_without_model_returns_400(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, fake_client, registry = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets",
        json={"name": "kg", "description": "x", "target_rows": 20, "modality": "graph"},
    )
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 0})
    assert r.status_code == 400
    assert not fake_client.calls
