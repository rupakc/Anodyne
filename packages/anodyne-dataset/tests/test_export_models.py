from datetime import UTC, datetime
from uuid import uuid4

from anodyne_dataset.models import ExportArtifact


def test_export_artifact_defaults_created_at() -> None:
    before = datetime.now(UTC)
    artifact = ExportArtifact(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        version_id=uuid4(),
        format="csv",
        row_count=50,
        object_key="datasets/d/v/export.csv",
    )
    assert artifact.created_at >= before
    assert artifact.format == "csv"
    assert artifact.row_count == 50


def test_export_artifact_round_trips_through_json() -> None:
    artifact = ExportArtifact(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        version_id=uuid4(),
        format="parquet",
        row_count=0,
        object_key="datasets/d/v/export.parquet",
    )
    dumped = artifact.model_dump(mode="json")
    restored = ExportArtifact.model_validate(dumped)
    assert restored == artifact
