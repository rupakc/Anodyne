"""Offline test for the evaluation activities: in-memory fakes, no Docker/Ray/LLM.

Exercises `run_evaluation` end to end with the default sequential runner against
fake repositories and an in-memory S3-like client, asserting the report artifacts
are written and the run + per-expert results are persisted.
"""

from __future__ import annotations

import io
from uuid import UUID, uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    Modality,
    SemanticType,
)
from anodyne_evaluation.models import EvaluationRun, ExpertScore
from anodyne_workflows.evaluation_activities import (
    EvaluationActivityContext,
    configure_evaluation_activities,
    run_evaluation,
    set_eval_status,
)
from anodyne_workflows.evaluation_workflow import EvaluationInput


def _parquet(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    n = 150
    x = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": rng.normal(2, 1, n),
            "label": (x > 0).astype(int),
            "group": rng.choice(["a", "b"], n),
        }
    )
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    return buf.getvalue()


class _FakeS3:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body):  # type: ignore[no-untyped-def]
        self.store[Key] = Body

    def get_object(self, *, Bucket, Key):  # type: ignore[no-untyped-def]
        return {"Body": io.BytesIO(self.store[Key])}


class _FakeDatasetRepo:
    def __init__(self, tenant: UUID, dataset: UUID, versions: list[DatasetVersion]) -> None:
        self._versions = versions
        self._spec = DatasetSpec(
            id=dataset,
            tenant_id=tenant,
            name="d",
            description="synthetic customers",
            modality=Modality.TABULAR,
            source="sample",
            fields=[FieldSpec(name="x", semantic_type=SemanticType.FLOAT)],
            target_rows=150,
        )

    async def list_versions(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._versions

    async def get_spec(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return self._spec


class _FakeEvalRepo:
    def __init__(self) -> None:
        self.runs: dict[UUID, EvaluationRun] = {}
        self.results: dict[UUID, list[ExpertScore]] = {}

    async def create_run(self, run):  # type: ignore[no-untyped-def]
        self.runs[run.id] = run

    async def save_run(self, run):  # type: ignore[no-untyped-def]
        self.runs[run.id] = run

    async def get_run(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return self.runs.get(run_id)

    async def list_runs(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return list(self.runs.values())

    async def add_expert_results(self, tenant_id, run_id, scores):  # type: ignore[no-untyped-def]
        self.results[run_id] = scores

    async def get_expert_results(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return self.results.get(run_id, [])


async def test_run_evaluation_writes_report_and_persists() -> None:
    tenant, dataset = uuid4(), uuid4()
    subj = DatasetVersion(
        id=uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/v1.parquet",
        format="parquet",
        row_count=150,
    )
    ref = DatasetVersion(
        id=uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/v2.parquet",
        format="parquet",
        row_count=150,
    )
    s3 = _FakeS3()
    # Keys are tenant-prefixed by S3ObjectStore.
    s3.store[f"{tenant}/{subj.artifact_uri}"] = _parquet(1)
    s3.store[f"{tenant}/{ref.artifact_uri}"] = _parquet(2)

    eval_repo = _FakeEvalRepo()
    ctx = EvaluationActivityContext(
        repo=eval_repo,  # type: ignore[arg-type]
        dataset_repo=_FakeDatasetRepo(tenant, dataset, [subj, ref]),  # type: ignore[arg-type]
        s3_bucket="anodyne",
        s3_client=s3,
    )
    configure_evaluation_activities(ctx)

    run_id = uuid4()
    inp = EvaluationInput(
        run_id=str(run_id),
        dataset_id=str(dataset),
        tenant_id=str(tenant),
        dataset_version_id=str(subj.id),
        reference_version_id=str(ref.id),
        seed=5,
        config={"target_field": "label", "sensitive_field": "group"},
    )
    await set_eval_status(inp, "running", 0.1)
    key = await run_evaluation(inp)

    assert key == f"evaluations/{run_id}/report.json"
    assert f"{tenant}/evaluations/{run_id}/report.json" in s3.store
    assert f"{tenant}/evaluations/{run_id}/report.html" in s3.store
    saved = eval_repo.runs[run_id]
    assert saved.overall_score is not None
    assert saved.report_html_uri == f"evaluations/{run_id}/report.html"
    # Statistical experts (no LLM configured) all applied with reference+target.
    dims = {str(s.dimension) for s in eval_repo.results[run_id]}
    assert {"fidelity", "privacy", "utility", "bias", "diversity"} <= dims
