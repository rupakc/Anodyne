"""Real activity implementations for `PerturbationWorkflow`.

Mirrors `anodyne_workflows.activities`: bound to infra via a module-level
context the worker sets once at startup; kept thin. The three activities load
the parent artifact, apply the injected `Perturbator` (a `RegistryPerturbator`,
which dispatches on modality through the perturbation registry), upload the
derived artifact, and register it as a new `DatasetVersion` carrying
`parent_version_id` -- matching how the generation activities register versions
and set status.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion, JobStatus, PerturbationFamily, PerturbationSpec
from anodyne_dataset.ports import DatasetRepository, PerturbationRepository, Perturbator
from anodyne_storage.objectstore import S3ObjectStore
from temporalio import activity

from anodyne_workflows.perturbation_workflow import PerturbationInput


class ProgressPublisher(Protocol):
    async def publish(self, channel: str, message: str) -> None: ...


@dataclass
class PerturbationActivityContext:
    """Infra bound to these activities by the worker at startup.

    `repo` and `perturbation_repo` are the same `SqlDatasetRepository` instance
    in production (it implements both roles); split here so tests can substitute
    focused fakes. `perturbator` is a `RegistryPerturbator` -- injected rather
    than imported so `perturbation_activities` needs no compile-time dependency
    on a concrete modality implementation.
    """

    repo: DatasetRepository
    perturbation_repo: PerturbationRepository
    perturbator: Perturbator
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None


_ctx: PerturbationActivityContext | None = None


def configure_perturbation_activities(ctx: PerturbationActivityContext) -> None:
    """Bind these activities to infra. Called once by the worker at startup."""
    global _ctx
    _ctx = ctx


def _context() -> PerturbationActivityContext:
    if _ctx is None:
        raise RuntimeError(
            "anodyne_workflows.perturbation_activities not configured: call "
            "configure_perturbation_activities() first"
        )
    return _ctx


def _object_store(inp: PerturbationInput) -> ObjectStore:
    ctx = _context()
    return S3ObjectStore(ctx.s3_bucket, uuid.UUID(inp.tenant_id), client=ctx.s3_client)


def _artifact_key(inp: PerturbationInput, ext: str) -> str:
    return f"datasets/{inp.dataset_id}/perturbations/{inp.job_id}/artifact.{ext}"


def _spec_from(inp: PerturbationInput) -> PerturbationSpec:
    return PerturbationSpec(
        family=PerturbationFamily(inp.family),
        intensity=inp.intensity,
        target_fields=list(inp.target_fields),
        params=dict(inp.params),
    )


def _read_table(data: bytes, fmt: str) -> pa.Table:
    if fmt == "parquet":
        return pq.read_table(io.BytesIO(data))
    if fmt == "jsonl":
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        return pa.Table.from_pylist(rows)
    raise ValueError(
        f"perturbation cannot read artifact format {fmt!r}; only parquet/jsonl "
        "(tabular/text) are supported"
    )


def _write_table(table: pa.Table, fmt: str) -> tuple[bytes, str]:
    if fmt == "parquet":
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue(), "parquet"
    # jsonl
    payload = "\n".join(json.dumps(row) for row in table.to_pylist())
    return payload.encode(), "jsonl"


async def _parent_version(inp: PerturbationInput) -> DatasetVersion:
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    versions = await ctx.repo.list_versions(tenant_id, uuid.UUID(inp.dataset_id))
    parent_id = uuid.UUID(inp.parent_version_id)
    parent = next((v for v in versions if v.id == parent_id), None)
    if parent is None:
        raise ValueError(
            f"parent version {inp.parent_version_id} not found for dataset {inp.dataset_id}"
        )
    return parent


@activity.defn(name="apply_perturbation")
async def apply_perturbation(inp: PerturbationInput) -> list[Any]:
    """Load the parent artifact, perturb it, upload the derived artifact.

    Returns `[uri, row_count]` (a list, so it survives Temporal's JSON payload
    round-trip cleanly). The heavy CPU transform runs via the injected
    `Perturbator`; a whole-artifact transform is a single step, so it runs
    in-activity rather than fanning out on Ray (a documented future seam).
    """
    ctx = _context()
    store = _object_store(inp)
    parent = await _parent_version(inp)
    data = await store.get(parent.artifact_uri)
    table = _read_table(data, parent.format)
    spec = _spec_from(inp)
    out = ctx.perturbator.perturb(spec, table, inp.modality, inp.seed)
    payload, ext = _write_table(out, parent.format)
    key = _artifact_key(inp, ext)
    await store.put(key, payload)
    return [key, out.num_rows]


@activity.defn(name="register_perturbed_version")
async def register_perturbed_version(inp: PerturbationInput, uri: str, rows: int) -> None:
    """Record the derived artifact as a new `DatasetVersion` (with lineage) and
    stamp `result_version_id` on the job -- matching generation's register step."""
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    parent = await _parent_version(inp)
    version = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        dataset_id=uuid.UUID(inp.dataset_id),
        artifact_uri=uri,
        format=parent.format,
        row_count=rows,
        parent_version_id=parent.id,
    )
    await ctx.repo.add_version(version)
    job = await ctx.perturbation_repo.get_perturbation_job(tenant_id, uuid.UUID(inp.job_id))
    if job is not None:
        job.result_version_id = version.id
        await ctx.perturbation_repo.save_perturbation_job(job)


@activity.defn(name="set_perturbation_status")
async def set_perturbation_status(
    inp: PerturbationInput, status: str, progress: float, message: str | None = None
) -> None:
    """Update the `PerturbationJob` status and publish live progress to Redis.

    Like generation's `set_status`, this fetches the existing job and mutates it
    in place (`save_perturbation_job` is a full upsert) so fields the gateway set
    at creation aren't wiped on a status transition.
    """
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    job_id = uuid.UUID(inp.job_id)
    job = await ctx.perturbation_repo.get_perturbation_job(tenant_id, job_id)
    if job is not None:
        job.status = JobStatus(status)
        job.progress = progress
        if message is not None:
            job.message = message
        await ctx.perturbation_repo.save_perturbation_job(job)
    if ctx.publisher is not None:
        payload = json.dumps({"job_id": inp.job_id, "status": status, "progress": progress})
        await ctx.publisher.publish(f"perturbation:{inp.job_id}", payload)


# Import the perturbation handlers for their registration side effect, mirroring
# how `activities` imports `handlers`. A `RegistryPerturbator` is what the worker
# injects as `ctx.perturbator`; importing here guarantees the registry is
# populated even if some future caller builds the perturbator lazily.
import anodyne_perturbation.handlers as _pert_handlers  # noqa: E402,F401
