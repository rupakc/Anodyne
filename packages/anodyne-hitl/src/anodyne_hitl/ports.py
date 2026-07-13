"""HITL ports (the hexagonal boundary for sub-system G).

Kept adapter-free, mirroring `anodyne_evaluation.ports`: no `temporalio` or SQL
import here. `anodyne-core` imports nothing from this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from anodyne_hitl.models import Annotation, Feedback, ReviewStatus, ReviewTask


class AnnotationRepository(ABC):
    @abstractmethod
    async def add(self, annotation: Annotation) -> None: ...

    @abstractmethod
    async def list_for_version(
        self, tenant_id: UUID, dataset_id: UUID, version_id: UUID
    ) -> list[Annotation]: ...

    @abstractmethod
    async def delete(self, tenant_id: UUID, annotation_id: UUID) -> bool:
        """Delete the annotation if it belongs to `tenant_id`. Returns whether
        a row was deleted (`False` means not found / not owned by this tenant)."""


class FeedbackRepository(ABC):
    @abstractmethod
    async def add(self, feedback: Feedback) -> None: ...

    @abstractmethod
    async def list_for_target(
        self, tenant_id: UUID, target_type: str, target_id: UUID
    ) -> list[Feedback]: ...


class ReviewRepository(ABC):
    @abstractmethod
    async def create(self, task: ReviewTask) -> None: ...

    @abstractmethod
    async def get(self, tenant_id: UUID, review_id: UUID) -> ReviewTask | None: ...

    @abstractmethod
    async def list_for_tenant(
        self, tenant_id: UUID, status: ReviewStatus | None = None
    ) -> list[ReviewTask]: ...

    @abstractmethod
    async def save(self, task: ReviewTask) -> None:
        """Persist a decided (or otherwise updated) review task."""
