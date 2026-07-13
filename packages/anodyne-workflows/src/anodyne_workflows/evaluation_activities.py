"""Activity implementations for `EvaluationWorkflow`.

Bind to infra via a module-level context set once by the evaluation worker
(mirrors `anodyne_workflows.activities`). `run_evaluation` is the one work
activity: it loads the version artifact(s) from the object store, builds the
mixture of experts (resolving the tenant's LLM for the qualitative judge through
the `LLMProvider` port), runs them via the injected `JudgeRunner` (Ray in
production, sequential otherwise), aggregates into a 360-degree report, renders
JSON + HTML, uploads both tenant-prefixed, and persists the run + per-expert
results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd  # type: ignore[import-untyped]
from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider, ObjectStore
from anodyne_dataset.models import DatasetVersion, Modality
from anodyne_dataset.ports import DatasetRepository
from anodyne_evaluation.evaluator import MoEEvaluator, judges_for_modality, sequential_runner
from anodyne_evaluation.loader import load_artifact, load_graph
from anodyne_evaluation.models import EvaluationConfig, EvaluationRun, EvaluationStatus
from anodyne_evaluation.ports import EvaluationContext, EvaluationRepository, JudgeRunner
from anodyne_evaluation.registry import (
    SqlEvaluationRepository,  # noqa: F401  (re-exported for wiring)
)
from anodyne_evaluation.report import render_html, render_json
from anodyne_storage.objectstore import S3ObjectStore
from temporalio import activity

from anodyne_workflows.evaluation_workflow import EvaluationInput

if TYPE_CHECKING:
    from anodyne_dataset.models import DatasetSpec


class ModelRegistryLike(Protocol):
    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> ModelConfig | None: ...

    async def list(self, tenant_id: uuid.UUID) -> list[ModelConfig]: ...


@dataclass
class EvaluationActivityContext:
    """Infra bound to the evaluation activities by the worker at startup."""

    repo: EvaluationRepository
    dataset_repo: DatasetRepository
    s3_bucket: str
    s3_client: object
    # Optional LLM wiring for the qualitative (LLM-as-a-Judge) expert. Absent =>
    # the qualitative dimension is simply skipped (statistical experts still run).
    llm_provider: LLMProvider | None = None
    model_registry: ModelRegistryLike | None = None
    # Parallel fan-out strategy; defaults to sequential (offline/no-Ray).
    runner: JudgeRunner = sequential_runner
    # Publish live progress (Redis), same duck type as generation's publisher.
    publisher: object | None = None


_ctx: EvaluationActivityContext | None = None


def configure_evaluation_activities(ctx: EvaluationActivityContext) -> None:
    global _ctx
    _ctx = ctx


def _context() -> EvaluationActivityContext:
    if _ctx is None:
        raise RuntimeError(
            "anodyne_workflows.evaluation_activities not configured: "
            "call configure_evaluation_activities() first"
        )
    return _ctx


def _object_store(ctx: EvaluationActivityContext, tenant_id: uuid.UUID) -> ObjectStore:
    return S3ObjectStore(ctx.s3_bucket, tenant_id, client=ctx.s3_client)


async def _load_version(
    ctx: EvaluationActivityContext,
    tenant_id: uuid.UUID,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> DatasetVersion:
    versions = await ctx.dataset_repo.list_versions(tenant_id, dataset_id)
    version = next((v for v in versions if v.id == version_id), None)
    if version is None:
        raise ValueError(f"dataset version {version_id} not found for tenant {tenant_id}")
    return version


@activity.defn(name="run_evaluation")
async def run_evaluation(inp: EvaluationInput) -> str:
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    dataset_id = uuid.UUID(inp.dataset_id)
    version_id = uuid.UUID(inp.dataset_version_id)
    cfg = EvaluationConfig.model_validate(inp.config or {})

    store = _object_store(ctx, tenant_id)
    version = await _load_version(ctx, tenant_id, dataset_id, version_id)
    ref_id = uuid.UUID(inp.reference_version_id) if inp.reference_version_id else None

    spec: DatasetSpec | None = await ctx.dataset_repo.get_spec(tenant_id, dataset_id)
    modality = spec.modality if spec else Modality.TABULAR

    # Graph versions are node-link JSON (not columnar): load them into a
    # `GraphDataset` for the graph judges and leave `subject` an empty frame.
    # Every other modality keeps the DataFrame path.
    subject: pd.DataFrame = pd.DataFrame()
    reference: pd.DataFrame | None = None
    subject_graph = None
    reference_graph = None
    if modality == Modality.GRAPH:
        subject_graph = load_graph(await store.get(version.artifact_uri))
        if ref_id is not None:
            ref_version = await _load_version(ctx, tenant_id, dataset_id, ref_id)
            reference_graph = load_graph(await store.get(ref_version.artifact_uri))
    else:
        subject = load_artifact(await store.get(version.artifact_uri), version.format)
        if ref_id is not None:
            ref_version = await _load_version(ctx, tenant_id, dataset_id, ref_id)
            reference = load_artifact(await store.get(ref_version.artifact_uri), ref_version.format)

    eval_ctx = EvaluationContext(
        subject=subject,
        reference=reference,
        modality=modality,
        sensitive_field=cfg.sensitive_field,
        target_field=cfg.target_field,
        text_column=cfg.text_column,
        sample_rows=cfg.sample_rows,
        seed=inp.seed or cfg.seed,
        metadata={"description": spec.description if spec else ""},
        subject_graph=subject_graph,
        reference_graph=reference_graph,
    )

    provider, model_cfg = await _resolve_llm(ctx, tenant_id, cfg)
    evaluator = MoEEvaluator(judges_for_modality(modality, provider, model_cfg), runner=ctx.runner)
    report = await evaluator.evaluate(
        eval_ctx,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        dataset_version_id=version_id,
        reference_version_id=ref_id,
        weights=cfg.weights or None,
    )

    json_key = f"evaluations/{inp.run_id}/report.json"
    html_key = f"evaluations/{inp.run_id}/report.html"
    await store.put(json_key, render_json(report))
    await store.put(html_key, render_html(report).encode("utf-8"))

    run = await ctx.repo.get_run(tenant_id, uuid.UUID(inp.run_id))
    if run is None:
        run = EvaluationRun(
            id=uuid.UUID(inp.run_id),
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            reference_version_id=ref_id,
        )
    run.report_uri = json_key
    run.report_html_uri = html_key
    run.overall_score = report.overall_score
    await ctx.repo.save_run(run)
    await ctx.repo.add_expert_results(tenant_id, run.id, report.expert_scores)
    return json_key


async def _resolve_llm(
    ctx: EvaluationActivityContext, tenant_id: uuid.UUID, cfg: EvaluationConfig
) -> tuple[LLMProvider | None, ModelConfig | None]:
    """Resolve the model driving the qualitative judge, or (None, None) to skip it."""
    if ctx.llm_provider is None or ctx.model_registry is None:
        return None, None
    model_cfg: ModelConfig | None = None
    if cfg.model_config_id is not None:
        model_cfg = await ctx.model_registry.get(tenant_id, cfg.model_config_id)
    else:
        configs = await ctx.model_registry.list(tenant_id)
        model_cfg = configs[0] if configs else None
    if model_cfg is None:
        return None, None
    return ctx.llm_provider, model_cfg


@activity.defn(name="set_eval_status")
async def set_eval_status(inp: EvaluationInput, status: str, progress: float) -> None:
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    run_id = uuid.UUID(inp.run_id)
    run = await ctx.repo.get_run(tenant_id, run_id)
    if run is None:
        run = EvaluationRun(
            id=run_id,
            tenant_id=tenant_id,
            dataset_id=uuid.UUID(inp.dataset_id),
            dataset_version_id=uuid.UUID(inp.dataset_version_id),
        )
    run.status = EvaluationStatus(status)
    run.progress = progress
    await ctx.repo.save_run(run)
