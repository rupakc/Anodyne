from uuid import uuid4

from anodyne_dataset.models import ColumnProfile, Profile, SemanticType


def test_column_profile_defaults() -> None:
    c = ColumnProfile(name="age", semantic_type=SemanticType.INTEGER)
    assert c.nullable is False
    assert c.null_rate == 0.0
    assert c.categories is None
    assert c.min is None and c.max is None


def test_profile_round_trip() -> None:
    tenant_id, dataset_id = uuid4(), uuid4()
    profile = Profile(
        id=uuid4(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        row_count=10,
        columns=[
            ColumnProfile(
                name="age", semantic_type=SemanticType.INTEGER, min=0.0, max=99.0, mean=42.0
            ),
            ColumnProfile(
                name="plan",
                semantic_type=SemanticType.CATEGORICAL,
                categories={"gold": 0.6, "silver": 0.4},
            ),
        ],
        correlations={"age": {"age": 1.0}},
        sample_uri="datasets/x/sample/data.csv",
        sample_filename="data.csv",
    )

    dumped = profile.model_dump(mode="json")
    restored = Profile.model_validate(dumped)

    assert restored == profile
    assert restored.columns[1].categories == {"gold": 0.6, "silver": 0.4}
    assert restored.sample_filename == "data.csv"
