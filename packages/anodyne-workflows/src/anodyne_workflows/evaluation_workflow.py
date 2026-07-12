"""Temporal workflow for the MoE LLM-as-a-Judge evaluation (sub-system F).

Mirrors `GenerationWorkflow`'s cadence (set-status around the work) but has no
HITL gate: an evaluation is a read-only benchmarking pass. The single
`run_evaluation` activity does the heavy lifting (load artifacts, fan the
experts out via Ray, aggregate, render + upload the report, persist results);
the workflow itself stays thin and imports no evaluation/adapter packages so it
remains deterministic per Temporal's sandbox rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


@dataclass
class EvaluationInput:
    run_id: str
    dataset_id: str
    tenant_id: str
    dataset_version_id: str
    reference_version_id: str | None = None
    seed: int = 0
    # Carried through to `run_evaluation`; the workflow never inspects it.
    config: dict[str, Any] = field(default_factory=dict)


@workflow.defn
class EvaluationWorkflow:
    @workflow.run
    async def run(self, inp: EvaluationInput) -> str:
        opts: dict[str, Any] = dict(
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        await workflow.execute_activity("set_eval_status", args=[inp, "running", 0.1], **opts)
        try:
            report_key: str = await workflow.execute_activity("run_evaluation", args=[inp], **opts)
        except Exception:
            await workflow.execute_activity("set_eval_status", args=[inp, "failed", 1.0], **opts)
            raise
        await workflow.execute_activity("set_eval_status", args=[inp, "succeeded", 1.0], **opts)
        return report_key
