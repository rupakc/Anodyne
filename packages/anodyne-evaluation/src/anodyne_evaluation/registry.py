"""SQL-backed `EvaluationRepository`.

Mirrors `anodyne_storage.dataset_repo.SqlDatasetRepository`: every method runs
inside a `tenant_session` (RLS `app.tenant_id` GUC via `SET LOCAL`), and reads
add an explicit `tenant_id` filter as defense-in-depth on top of RLS. Lives in
the evaluation package (not in anodyne-storage), matching how
`SqlImageProviderRegistry` lives in anodyne-image while using the shared
`anodyne_storage.db` tables.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from anodyne_storage.db import (
    evaluation_expert_results,
    evaluation_runs,
    tenant_session,
)
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from anodyne_evaluation.models import EvalDimension, EvaluationRun, EvaluationStatus, ExpertScore
from anodyne_evaluation.ports import EvaluationRepository


def _run_from_row(m: Any) -> EvaluationRun:
    return EvaluationRun(
        id=m["id"],
        tenant_id=m["tenant_id"],
        dataset_id=m["dataset_id"],
        dataset_version_id=m["dataset_version_id"],
        reference_version_id=m["reference_version_id"],
        status=EvaluationStatus(m["status"]),
        progress=m["progress"],
        message=m["message"],
        workflow_id=m["workflow_id"],
        report_uri=m["report_uri"],
        report_html_uri=m["report_html_uri"],
        overall_score=m["overall_score"],
        config=m["config"],
        created_at=m["created_at"],
    )


def _score_from_row(m: Any) -> ExpertScore:
    return ExpertScore(
        dimension=EvalDimension(m["dimension"]),
        score=m["score"],
        rationale=m["rationale"],
        metrics=m["metrics"],
        recommendations=m["recommendations"],
    )


class SqlEvaluationRepository(EvaluationRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def _values(self, run: EvaluationRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "tenant_id": run.tenant_id,
            "dataset_id": run.dataset_id,
            "dataset_version_id": run.dataset_version_id,
            "reference_version_id": run.reference_version_id,
            "status": str(run.status),
            "progress": run.progress,
            "message": run.message,
            "workflow_id": run.workflow_id,
            "report_uri": run.report_uri,
            "report_html_uri": run.report_html_uri,
            "overall_score": run.overall_score,
            "config": run.config,
            "created_at": run.created_at,
        }

    async def create_run(self, run: EvaluationRun) -> None:
        await self.save_run(run)

    async def save_run(self, run: EvaluationRun) -> None:
        values = self._values(run)
        async with tenant_session(self._engine, run.tenant_id) as s:
            stmt = pg_insert(evaluation_runs).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[evaluation_runs.c.id],
                set_={k: v for k, v in values.items() if k != "id"},
            )
            await s.execute(stmt)
            await s.commit()

    async def get_run(self, tenant_id: UUID, run_id: UUID) -> EvaluationRun | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                (
                    await s.execute(
                        select(evaluation_runs).where(
                            evaluation_runs.c.id == run_id,
                            evaluation_runs.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
            return _run_from_row(row) if row else None

    async def list_runs(self, tenant_id: UUID, dataset_id: UUID) -> list[EvaluationRun]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(evaluation_runs).where(
                            evaluation_runs.c.dataset_id == dataset_id,
                            evaluation_runs.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_run_from_row(r) for r in rows]

    async def add_expert_results(
        self, tenant_id: UUID, run_id: UUID, scores: list[ExpertScore]
    ) -> None:
        if not scores:
            return
        async with tenant_session(self._engine, tenant_id) as s:
            # Replace any prior results for this run so a re-run is idempotent.
            await s.execute(
                delete(evaluation_expert_results).where(
                    evaluation_expert_results.c.run_id == run_id,
                    evaluation_expert_results.c.tenant_id == tenant_id,
                )
            )
            await s.execute(
                evaluation_expert_results.insert(),
                [
                    {
                        "id": uuid4(),
                        "tenant_id": tenant_id,
                        "run_id": run_id,
                        "dimension": str(sc.dimension),
                        "score": sc.score,
                        "rationale": sc.rationale,
                        "metrics": sc.metrics,
                        "recommendations": sc.recommendations,
                    }
                    for sc in scores
                ],
            )
            await s.commit()

    async def get_expert_results(self, tenant_id: UUID, run_id: UUID) -> list[ExpertScore]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(evaluation_expert_results).where(
                            evaluation_expert_results.c.run_id == run_id,
                            evaluation_expert_results.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_score_from_row(r) for r in rows]
