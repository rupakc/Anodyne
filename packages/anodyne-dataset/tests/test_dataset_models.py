from uuid import uuid4

from anodyne_dataset.models import (
    DatasetSpec,
    FieldSpec,
    GenerationJob,
    JobStatus,
    Modality,
    SemanticType,
)


def test_fieldspec_defaults() -> None:
    f = FieldSpec(name="age", semantic_type=SemanticType.INTEGER)
    assert f.nullable is False and f.constraints == {}


def test_datasetspec_is_tabular_description() -> None:
    spec = DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="people",
        modality=Modality.TABULAR,
        source="description",
        schema=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=100,
    )
    assert spec.status == "draft" and spec.schema[0].name == "age"


def test_job_progress_bounds() -> None:
    j = GenerationJob(id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4())
    assert j.status is JobStatus.PENDING and j.progress == 0.0
