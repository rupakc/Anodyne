"""Ray-backed `JudgeRunner`: fan the mixture of experts out over the cluster.

The CPU-bound `StatisticalJudge`s are dispatched as `@ray.remote` tasks (each
runs its synchronous `compute`); the async LLM-backed qualitative judge can't be
pickled/run on a Ray worker so it runs inline in the caller. Matches the
"Ray owns compute, Temporal owns flow" split: this is the compute fan-out used
inside the single `run_evaluation` Temporal activity.

Remote tasks return `ExpertScore | None` (None == the judge raised
`JudgeNotApplicable`) so a not-applicable expert never surfaces as a
`RayTaskError` on `ray.get`.
"""

from __future__ import annotations

from collections.abc import Sequence

import ray
from anodyne_evaluation.judges.base import StatisticalJudge
from anodyne_evaluation.models import ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable


@ray.remote
def _remote_compute(judge: StatisticalJudge, ctx: EvaluationContext) -> ExpertScore | None:
    try:
        return judge.compute(ctx)
    except JudgeNotApplicable:
        return None


class RayJudgeRunner:
    """A `JudgeRunner` that parallelizes the statistical experts via Ray."""

    async def __call__(self, judges: Sequence[Judge], ctx: EvaluationContext) -> list[ExpertScore]:
        statistical = [j for j in judges if isinstance(j, StatisticalJudge)]
        other = [j for j in judges if not isinstance(j, StatisticalJudge)]

        refs = [_remote_compute.remote(j, ctx) for j in statistical]
        scores: list[ExpertScore] = [s for s in ray.get(refs) if s is not None]

        for judge in other:  # qualitative (LLM) expert(s): async, run inline
            try:
                scores.append(await judge.evaluate(ctx))
            except JudgeNotApplicable:
                continue
        return scores
