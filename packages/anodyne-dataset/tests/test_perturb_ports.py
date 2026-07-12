from uuid import UUID, uuid4

from anodyne_dataset.models import PerturbationFamily, PerturbationJob, PerturbationSpec
from anodyne_dataset.ports import PerturbationRepository, Perturbator


def test_perturbator_is_abstract() -> None:
    assert getattr(Perturbator.perturb, "__isabstractmethod__", False) is True


async def test_perturbation_repository_roundtrip_via_fake() -> None:
    class _FakeRepo(PerturbationRepository):
        def __init__(self) -> None:
            self.jobs: dict[UUID, PerturbationJob] = {}

        async def save_perturbation_job(self, job: PerturbationJob) -> None:
            self.jobs[job.id] = job

        async def get_perturbation_job(
            self, tenant_id: UUID, job_id: UUID
        ) -> PerturbationJob | None:
            job = self.jobs.get(job_id)
            return job if job and job.tenant_id == tenant_id else None

        async def list_perturbation_jobs(
            self, tenant_id: UUID, dataset_id: UUID
        ) -> list[PerturbationJob]:
            return [
                j
                for j in self.jobs.values()
                if j.tenant_id == tenant_id and j.dataset_id == dataset_id
            ]

    repo = _FakeRepo()
    tenant = uuid4()
    dataset = uuid4()
    job = PerturbationJob(
        id=uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        parent_version_id=uuid4(),
        spec=PerturbationSpec(family=PerturbationFamily.NOISE),
    )
    await repo.save_perturbation_job(job)
    assert (await repo.get_perturbation_job(tenant, job.id)) == job
    assert await repo.get_perturbation_job(uuid4(), job.id) is None
    assert await repo.list_perturbation_jobs(tenant, dataset) == [job]
