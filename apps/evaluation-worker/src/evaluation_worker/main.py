"""Temporal worker process for `EvaluationWorkflow`.

Binds the evaluation activities in `anodyne_workflows.evaluation_activities` to
real infra (`SqlEvaluationRepository`, `SqlDatasetRepository`, `S3ObjectStore`
client, `RayJudgeRunner`, and -- when a `secret_key` is configured -- a
`LiteLLMProvider` + `SqlModelRegistry` for the qualitative LLM-as-a-Judge
expert), then runs a `temporalio` `Worker` on the "evaluation" task queue.
`build_worker` is the pure wiring step (testable with fakes, no live Temporal);
`main` is the entrypoint: ``python -m evaluation_worker.main``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import boto3  # type: ignore[import-untyped]
from anodyne_compute import ray_init
from anodyne_compute.ray_evaluation import RayJudgeRunner
from anodyne_core.ports import LLMProvider
from anodyne_dataset.ports import DatasetRepository
from anodyne_evaluation.ports import EvaluationRepository, JudgeRunner
from anodyne_evaluation.registry import SqlEvaluationRepository
from anodyne_llm.adapter import LiteLLMProvider
from anodyne_llm.registry import SqlModelRegistry
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
from anodyne_workflows.evaluation_activities import (
    EvaluationActivityContext,
    ModelRegistryLike,
    configure_evaluation_activities,
    run_evaluation,
    set_eval_status,
)
from anodyne_workflows.evaluation_workflow import EvaluationWorkflow
from temporalio.client import Client
from temporalio.worker import Worker

from evaluation_worker.config import get_settings

TASK_QUEUE = "evaluation"


@dataclass
class WorkerDeps:
    repo: EvaluationRepository
    dataset_repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    llm_provider: LLMProvider | None = None
    model_registry: ModelRegistryLike | None = None
    runner: JudgeRunner | None = None


def registered_workflows() -> list[type]:
    return [EvaluationWorkflow]


def registered_activities() -> list[Any]:
    return [run_evaluation, set_eval_status]


def build_worker(client: Client, deps: WorkerDeps) -> Worker:
    configure_evaluation_activities(
        EvaluationActivityContext(
            repo=deps.repo,
            dataset_repo=deps.dataset_repo,
            s3_bucket=deps.s3_bucket,
            s3_client=deps.s3_client,
            llm_provider=deps.llm_provider,
            model_registry=deps.model_registry,
            runner=deps.runner or RayJudgeRunner(),
        )
    )
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=registered_workflows(),
        activities=registered_activities(),
    )


async def main() -> None:
    settings = get_settings()
    ray_init(settings.ray_address)

    engine = make_engine(settings.database_url)
    repo = SqlEvaluationRepository(engine)
    dataset_repo = SqlDatasetRepository(engine)
    s3_client = boto3.client("s3")

    llm_provider: LLMProvider | None = None
    model_registry: SqlModelRegistry | None = None
    if settings.secret_key:
        secret_store = FernetSecretStore(settings.secret_key.encode())
        llm_provider = LiteLLMProvider(secret_store)
        model_registry = SqlModelRegistry(engine, secret_store)

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client,
        WorkerDeps(
            repo=repo,
            dataset_repo=dataset_repo,
            s3_bucket=settings.s3_bucket,
            s3_client=s3_client,
            llm_provider=llm_provider,
            model_registry=model_registry,
            runner=RayJudgeRunner(),
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
