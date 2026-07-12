from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_dataset.models import ColumnProfile, Profile, SemanticType
from anodyne_dataset.ports import ProfileRepository, SampleProfiler


class _FakeProfiler(SampleProfiler):
    def profile(
        self, tenant_id: UUID, dataset_id: UUID, sample_uri: str, data: bytes, filename: str
    ) -> Profile:
        return Profile(
            id=uuid4(),
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            row_count=0,
            columns=[ColumnProfile(name="x", semantic_type=SemanticType.INTEGER)],
            sample_uri=sample_uri,
            sample_filename=filename,
        )


class _FakeProfileRepository(ProfileRepository):
    def __init__(self) -> None:
        self.saved: dict[UUID, Profile] = {}

    async def save_profile(self, profile: Profile) -> None:
        self.saved[profile.dataset_id] = profile

    async def get_profile(self, tenant_id: UUID, dataset_id: UUID) -> Profile | None:
        p = self.saved.get(dataset_id)
        return p if p is not None and p.tenant_id == tenant_id else None


def test_sample_profiler_port_is_implementable() -> None:
    tenant_id, dataset_id = uuid4(), uuid4()
    profile = _FakeProfiler().profile(tenant_id, dataset_id, "k", b"data", "s.csv")
    assert profile.tenant_id == tenant_id
    assert profile.sample_filename == "s.csv"


async def test_profile_repository_port_is_implementable() -> None:
    repo = _FakeProfileRepository()
    tenant_id, dataset_id = uuid4(), uuid4()
    profile = Profile(
        id=uuid4(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        row_count=1,
        columns=[],
        sample_uri="k",
        sample_filename="f.csv",
    )

    await repo.save_profile(profile)

    assert (await repo.get_profile(tenant_id, dataset_id)) == profile
    assert (await repo.get_profile(uuid4(), dataset_id)) is None
