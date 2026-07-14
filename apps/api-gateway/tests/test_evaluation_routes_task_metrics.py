from __future__ import annotations

from uuid import uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import DatasetSpec, DatasetVersion, FieldSpec, Modality, SemanticType
from anodyne_evaluation.models import EvaluationRun
from anodyne_workflows.evaluation_workflow import EvaluationInput
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role = Role.MEMBER) -> TenantContext:
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])


class _FakeDatasetRepo:
    def __init__(
        self, versions: list[DatasetVersion], specs: dict[object, DatasetSpec] | None = None
    ) -> None:
        self._versions = versions
        self._specs = specs or {}

    async def list_versions(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._versions

    async def get_version(self, tenant_id, version_id):  # type: ignore[no-untyped-def]
        for v in self._versions:
            if v.id == version_id and v.tenant_id == tenant_id:
                return v
        return None

    async def get_spec(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        spec = self._specs.get(dataset_id)
        if spec is not None and spec.tenant_id == tenant_id:
            return spec
        return None


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
        self.last_input: EvaluationInput | None = None

    async def start_workflow(self, run_fn, input_arg, *a, **k):  # type: ignore[no-untyped-def]
        self.started = True
        self.last_input = input_arg
        return _Handle()


class _FakeEmptyModelRegistry:
    async def list(self, tenant_id):  # type: ignore[no-untyped-def]
        return []


def _text_spec(tenant_id: object, dataset_id: object) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,  # type: ignore[arg-type]
        tenant_id=tenant_id,  # type: ignore[arg-type]
        name="ds",
        description="d",
        modality=Modality.TEXT,
        source="synthetic",
        fields=[
            FieldSpec(name="text", semantic_type=SemanticType.TEXT),
            FieldSpec(name="label", semantic_type=SemanticType.CATEGORICAL),
        ],
        target_rows=10,
    )


def _tabular_spec(tenant_id: object, dataset_id: object) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,  # type: ignore[arg-type]
        tenant_id=tenant_id,  # type: ignore[arg-type]
        name="ds",
        description="d",
        modality=Modality.TABULAR,
        source="synthetic",
        fields=[
            FieldSpec(name="age", semantic_type=SemanticType.INTEGER),
            FieldSpec(name="income", semantic_type=SemanticType.FLOAT),
            FieldSpec(name="churned", semantic_type=SemanticType.CATEGORICAL),
        ],
        target_rows=10,
    )


@pytest.fixture
def client_and_app():  # type: ignore[no-untyped-def]
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app


async def test_task_metrics_catalog_for_text_classification(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    dataset_id, version_id = uuid4(), uuid4()
    version = DatasetVersion(
        id=version_id,
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        artifact_uri="a",
        format="parquet",
        row_count=10,
    )
    spec = _text_spec(ctx.tenant_id, dataset_id)
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo(
        [version], {dataset_id: spec}
    )

    r = await client.get(f"/datasets/{dataset_id}/versions/{version_id}/task-metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["task_type"] == "text_classification"
    assert body["available_metrics"]
    for m in body["available_metrics"]:
        assert set(["key", "label", "description", "requires_llm"]).issubset(m.keys())


async def test_task_metrics_tabular_categorical_target_resolves_classification(  # type: ignore[no-untyped-def]
    client_and_app,
) -> None:
    """A tabular spec with a categorical target + `?target_field=` must resolve to
    `tabular_classification` and its tabular catalog -- matching what
    `evaluation_activities.run_evaluation` resolves at evaluation time, so the
    catalog the UI offers is never a dead-letter superset/subset of the real one.
    """
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    dataset_id, version_id = uuid4(), uuid4()
    version = DatasetVersion(
        id=version_id,
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        artifact_uri="a",
        format="parquet",
        row_count=10,
    )
    spec = _tabular_spec(ctx.tenant_id, dataset_id)
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo(
        [version], {dataset_id: spec}
    )

    r = await client.get(
        f"/datasets/{dataset_id}/versions/{version_id}/task-metrics?target_field=churned"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task_type"] == "tabular_classification"
    assert body["available_metrics"]
    for m in body["available_metrics"]:
        assert set(["key", "label", "description", "requires_llm"]).issubset(m.keys())


async def test_task_metrics_tabular_numeric_target_resolves_regression(  # type: ignore[no-untyped-def]
    client_and_app,
) -> None:
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    dataset_id, version_id = uuid4(), uuid4()
    version = DatasetVersion(
        id=version_id,
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        artifact_uri="a",
        format="parquet",
        row_count=10,
    )
    spec = _tabular_spec(ctx.tenant_id, dataset_id)
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo(
        [version], {dataset_id: spec}
    )

    r = await client.get(
        f"/datasets/{dataset_id}/versions/{version_id}/task-metrics?target_field=income"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task_type"] == "regression"


async def test_task_metrics_tabular_without_target_field_still_generic(  # type: ignore[no-untyped-def]
    client_and_app,
) -> None:
    """Documents that the drift is closed only when `target_field` is passed --
    omitting it (the pre-fix call shape) still falls back to `generic`, same as
    before."""
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    dataset_id, version_id = uuid4(), uuid4()
    version = DatasetVersion(
        id=version_id,
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        artifact_uri="a",
        format="parquet",
        row_count=10,
    )
    spec = _tabular_spec(ctx.tenant_id, dataset_id)
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo(
        [version], {dataset_id: spec}
    )

    r = await client.get(f"/datasets/{dataset_id}/versions/{version_id}/task-metrics")
    assert r.status_code == 200
    assert r.json()["task_type"] == "generic"


async def test_task_metrics_unknown_version_404(client_and_app) -> None:  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx(Role.VIEWER)
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo([], {})

    r = await client.get(f"/datasets/{uuid4()}/versions/{uuid4()}/task-metrics")
    assert r.status_code == 404


async def test_evaluate_passes_selected_metrics_and_task_type(client_and_app) -> None:  # type: ignore[no-untyped-def]
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
        json={"selected_metrics": ["accuracy"], "task_type": "text_classification"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["config"]["selected_metrics"] == ["accuracy"]
    assert body["config"]["task_type"] == "text_classification"
    assert fake_client.last_input is not None
    assert fake_client.last_input.config["selected_metrics"] == ["accuracy"]
    assert fake_client.last_input.config["task_type"] == "text_classification"


async def test_evaluate_rejects_invalid_task_type(client_and_app) -> None:  # type: ignore[no-untyped-def]
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
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_dataset_repo] = lambda: _FakeDatasetRepo([version])
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: _FakeEvalRepo()
    app.dependency_overrides[deps.get_temporal_client] = lambda: _FakeClient()
    app.dependency_overrides[deps.get_model_registry] = lambda: _FakeEmptyModelRegistry()

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/evaluate",
        json={"task_type": "nonsense"},
    )
    assert r.status_code == 422
