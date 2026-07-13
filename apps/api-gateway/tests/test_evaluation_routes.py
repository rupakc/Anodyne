from __future__ import annotations

import json
from uuid import uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import DatasetVersion
from anodyne_evaluation.models import EvaluationRun
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role = Role.MEMBER) -> TenantContext:
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])


class _FakeDatasetRepo:
    def __init__(self, versions: list[DatasetVersion]) -> None:
        self._versions = versions

    async def list_versions(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._versions


class _FakeEvalRepo:
    def __init__(self) -> None:
        self.runs: dict[object, EvaluationRun] = {}

    async def create_run(self, run):  # type: ignore[no-untyped-def]
        self.runs[run.id] = run

    async def get_run(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        r = self.runs.get(run_id)
        return r if (r and r.tenant_id == tenant_id) else None


class _Handle:
    id = "eval-wf-1"


class _FakeClient:
    def __init__(self) -> None:
        self.started = False

    async def start_workflow(self, *a, **k):  # type: ignore[no-untyped-def]
        self.started = True
        return _Handle()


class _FakeStore:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def get(self, key):  # type: ignore[no-untyped-def]
        return self._payload

    async def presigned_url(self, key, expires=3600):  # type: ignore[no-untyped-def]
        return f"https://signed/{key}"


class _FakeEmptyModelRegistry:
    """No models registered for the tenant -- exercises the "no crash, no
    default resolved" branch of `start_evaluation`'s default-model lookup."""

    async def list(self, tenant_id):  # type: ignore[no-untyped-def]
        return []


@pytest.fixture
def client_and_app():  # type: ignore[no-untyped-def]
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app


async def test_start_evaluation_starts_workflow(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx()
    dataset_id, version_id = uuid4(), uuid4()
    version = DatasetVersion(
        id=version_id,
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        artifact_uri="a",
        format="parquet",
        row_count=10,
    )
    eval_repo = _FakeEvalRepo()
    fake_client = _FakeClient()
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo([version])
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: eval_repo
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    app.dependency_overrides[deps.get_model_registry] = lambda: _FakeEmptyModelRegistry()

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/evaluate",
        json={"target_field": "label", "sensitive_field": "group"},
    )
    assert r.status_code == 202
    assert fake_client.started
    body = r.json()
    assert body["workflow_id"] == "eval-wf-1"
    assert body["config"]["target_field"] == "label"
    assert eval_repo.runs  # persisted


async def test_start_evaluation_rejects_unknown_version(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx()
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo([])
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: _FakeEvalRepo()
    app.dependency_overrides[deps.get_temporal_client] = lambda: _FakeClient()
    app.dependency_overrides[deps.get_model_registry] = lambda: _FakeEmptyModelRegistry()
    r = await client.post(f"/datasets/{uuid4()}/versions/{uuid4()}/evaluate", json={})
    assert r.status_code == 404


async def test_evaluate_requires_write_permission(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)
    r = await client.post(f"/datasets/{uuid4()}/versions/{uuid4()}/evaluate", json={})
    assert r.status_code == 403


async def test_get_report_and_download(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    run = EvaluationRun(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        report_uri="evaluations/r/report.json",
        report_html_uri="evaluations/r/report.html",
        overall_score=0.7,
    )
    eval_repo = _FakeEvalRepo()
    eval_repo.runs[run.id] = run
    payload = json.dumps({"overall_score": 0.7, "summary": "ok"}).encode()
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: eval_repo
    app.dependency_overrides[deps.get_object_store] = lambda: _FakeStore(payload)

    status = await client.get(f"/evaluations/{run.id}")
    assert status.status_code == 200
    assert status.json()["overall_score"] == 0.7

    report = await client.get(f"/evaluations/{run.id}/report")
    assert report.status_code == 200
    assert report.json()["summary"] == "ok"

    # No presigned URL: the report bytes are streamed straight back through
    # the gateway with a Content-Disposition attachment header.
    dl_html = await client.get(f"/evaluations/{run.id}/report/download")
    assert dl_html.status_code == 200
    assert dl_html.headers["content-type"].startswith("text/html")
    assert (
        dl_html.headers["content-disposition"] == f'attachment; filename="evaluation-{run.id}.html"'
    )
    assert dl_html.content == payload

    dl_json = await client.get(f"/evaluations/{run.id}/report/download?format=json")
    assert dl_json.status_code == 200
    assert dl_json.headers["content-type"] == "application/json"
    assert (
        dl_json.headers["content-disposition"] == f'attachment; filename="evaluation-{run.id}.json"'
    )
    assert dl_json.content == payload


async def test_get_report_404_for_other_tenant(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    run = EvaluationRun(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        report_uri="k",
    )
    eval_repo = _FakeEvalRepo()
    eval_repo.runs[run.id] = run  # belongs to a different tenant
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: eval_repo
    r = await client.get(f"/evaluations/{run.id}")
    assert r.status_code == 404
