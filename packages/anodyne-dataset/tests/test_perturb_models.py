from uuid import uuid4

from anodyne_dataset.models import (
    DatasetVersion,
    JobStatus,
    PerturbationFamily,
    PerturbationJob,
    PerturbationSpec,
)


def test_perturbation_spec_defaults() -> None:
    spec = PerturbationSpec(family=PerturbationFamily.NOISE)
    assert spec.intensity == 0.1
    assert spec.target_fields == []
    assert spec.params == {}


def test_perturbation_family_values() -> None:
    assert {f.value for f in PerturbationFamily} == {
        "noise",
        "drift",
        "outliers",
        "bias",
        "edge_case",
        "graph_rewire",
        "graph_dropout",
        "graph_ontology_violation",
    }


def test_perturbation_job_defaults() -> None:
    job = PerturbationJob(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        parent_version_id=uuid4(),
        spec=PerturbationSpec(family=PerturbationFamily.BIAS, intensity=0.5),
    )
    assert job.status is JobStatus.PENDING
    assert job.progress == 0.0
    assert job.result_version_id is None
    assert job.spec.family is PerturbationFamily.BIAS


def test_dataset_version_lineage_optional() -> None:
    v = DatasetVersion(id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4(), artifact_uri="k")
    assert v.parent_version_id is None
    parent = uuid4()
    v2 = DatasetVersion(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        artifact_uri="k2",
        parent_version_id=parent,
    )
    assert v2.parent_version_id == parent
