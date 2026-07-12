"""Durable perturbation workflow: parent `DatasetVersion` in, derived one out.

Deliberately separate from `GenerationWorkflow` (a perturbation acts on an
already-generated version, not a spec) but structurally identical: orchestration
only, dispatching activities purely by **name** so this module -- like
`workflow.py` -- imports no adapter/modality package and stays Temporal-
deterministic. The per-modality corruption lives behind `apply_perturbation` in
the perturbation modality registry (`anodyne_perturbation`), exactly as the
generation activities hide per-modality generation behind the modality registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


@dataclass
class PerturbationInput:
    job_id: str
    dataset_id: str
    tenant_id: str
    parent_version_id: str
    family: str
    intensity: float
    seed: int
    # Additive; defaults keep every caller/test terse. The workflow never
    # inspects these -- they ride through to the activities like `seed`.
    params: dict[str, Any] = field(default_factory=dict)
    target_fields: list[str] = field(default_factory=list)
    modality: str = "tabular"


@workflow.defn
class PerturbationWorkflow:
    @workflow.run
    async def run(self, inp: PerturbationInput) -> str:
        opts: dict[str, Any] = dict(
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        await workflow.execute_activity(
            "set_perturbation_status", args=[inp, "running", 0.1], **opts
        )
        result: list[Any] = await workflow.execute_activity(
            "apply_perturbation", args=[inp], **opts
        )
        uri, rows = result[0], result[1]
        await workflow.execute_activity("register_perturbed_version", args=[inp, uri, rows], **opts)
        await workflow.execute_activity(
            "set_perturbation_status", args=[inp, "succeeded", 1.0], **opts
        )
        return str(uri)
