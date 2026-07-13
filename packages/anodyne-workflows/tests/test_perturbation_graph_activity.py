"""Offline end-to-end test of the graph-aware branch in `apply_perturbation`.

A parent version whose `format` is ``graph_json`` must be loaded via
`anodyne_graph.serialization.from_json_bytes`, perturbed by the graph
perturbator, and re-serialized via `to_json_bytes` -- never forced through a
`pyarrow.Table`. We also assert the injected ontology violations measurably drop
the `OntologyConsistencyGraphJudge` score (the semantic family's contract). No
Temporal/Ray/network/LLM.
"""

from __future__ import annotations

import uuid

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import (
    DatasetVersion,
    PerturbationFamily,
    PerturbationJob,
    PerturbationSpec,
)
from anodyne_evaluation.graph_judges.ontology import OntologyConsistencyGraphJudge
from anodyne_evaluation.ports import EvaluationContext
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
    compute_metrics,
)
from anodyne_graph.serialization import from_json_bytes, to_json_bytes
from anodyne_perturbation import RegistryPerturbator
from anodyne_workflows.perturbation_activities import (
    PerturbationActivityContext,
    apply_perturbation,
    configure_perturbation_activities,
)
from anodyne_workflows.perturbation_workflow import PerturbationInput


class _FakeStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.data[key] = data

    async def get(self, key: str) -> bytes:
        return self.data[key]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://x/{key}"

    async def list(self, prefix: str) -> list[str]:
        return [k for k in self.data if k.startswith(prefix)]


class _FakeRepo:
    def __init__(self) -> None:
        self.versions: dict[uuid.UUID, list[DatasetVersion]] = {}
        self.jobs: dict[uuid.UUID, PerturbationJob] = {}

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.setdefault(version.dataset_id, []).append(version)

    async def list_versions(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return [v for v in self.versions.get(dataset_id, []) if v.tenant_id == tenant_id]

    async def save_perturbation_job(self, job: PerturbationJob) -> None:
        self.jobs[job.id] = job

    async def get_perturbation_job(self, tenant_id, job_id):  # type: ignore[no-untyped-def]
        j = self.jobs.get(job_id)
        return j if j and j.tenant_id == tenant_id else None


def _dataset() -> GraphDataset:
    onto = GraphOntology(
        node_types=[
            NodeType(
                name="Person",
                properties=[
                    PropertySpec(
                        name="age", datatype="integer", constraints={"min": 0, "max": 120}
                    ),
                    PropertySpec(name="full_name", datatype="string"),
                ],
            ),
            NodeType(name="Company"),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )
    nodes = [
        Node(id=f"p{i}", type="Person", properties={"age": 30 + i, "full_name": f"Faked {i}"})
        for i in range(5)
    ]
    nodes += [Node(id=f"c{j}", type="Company") for j in range(2)]
    edges = [
        Edge(id=f"e{i}", type="WORKS_AT", source=f"p{i}", target=f"c{i % 2}") for i in range(5)
    ]
    return GraphDataset(
        ontology=onto, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def _judge_score(ds: GraphDataset) -> float:
    ctx = EvaluationContext(subject=pd.DataFrame(), subject_graph=ds)
    return OntologyConsistencyGraphJudge().compute(ctx).score


def _wire(family: PerturbationFamily, intensity: float):  # type: ignore[no-untyped-def]
    tenant, dataset = uuid.uuid4(), uuid.uuid4()
    store, repo = _FakeStore(), _FakeRepo()
    ds = _dataset()
    parent = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/gen/artifact.json",
        format="graph_json",
        row_count=len(ds.nodes),
    )
    store.data[parent.artifact_uri] = to_json_bytes(ds)
    repo.versions[dataset] = [parent]
    job = PerturbationJob(
        id=uuid.uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        parent_version_id=parent.id,
        spec=PerturbationSpec(family=family, intensity=intensity),
    )
    repo.jobs[job.id] = job
    configure_perturbation_activities(
        PerturbationActivityContext(
            repo=repo,  # type: ignore[arg-type]
            perturbation_repo=repo,  # type: ignore[arg-type]
            perturbator=RegistryPerturbator(),
            s3_bucket="b",
            s3_client=None,
        )
    )
    inp = PerturbationInput(
        job_id=str(job.id),
        dataset_id=str(dataset),
        tenant_id=str(tenant),
        parent_version_id=str(parent.id),
        family=family.value,
        intensity=intensity,
        seed=9,
        modality="graph",
    )
    return store, inp, ds


async def test_graph_artifact_round_trips_through_graph_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store, inp, parent_ds = _wire(PerturbationFamily.GRAPH_DROPOUT, 1.0)
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)
    uri, rows = await apply_perturbation(inp)

    assert uri in store.data
    out = from_json_bytes(store.data[uri])  # parses => it is valid graph_json
    assert out.edges == []  # dropout at intensity 1
    assert {n.id for n in out.nodes} == {n.id for n in parent_ds.nodes}
    assert rows == len(out.nodes)


async def test_graph_ontology_violation_drops_judge_score(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store, inp, parent_ds = _wire(PerturbationFamily.GRAPH_ONTOLOGY_VIOLATION, 1.0)
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)

    base_score = _judge_score(parent_ds)
    uri, _ = await apply_perturbation(inp)
    perturbed = from_json_bytes(store.data[uri])
    assert base_score == pytest.approx(1.0)
    assert _judge_score(perturbed) < base_score


async def test_graph_perturbation_intensity_zero_is_noop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    store, inp, parent_ds = _wire(PerturbationFamily.GRAPH_REWIRE, 0.0)
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)
    uri, _ = await apply_perturbation(inp)
    assert store.data[uri] == to_json_bytes(parent_ds)
