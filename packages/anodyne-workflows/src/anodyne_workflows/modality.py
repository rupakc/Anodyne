"""Per-modality dispatch registry for the shared generation activities.

The four shared Temporal activities in `anodyne_workflows.activities`
(`plan_shards`/`generate_shards`/`assemble_and_upload`/`register_version`)
own the orchestration shape common to *every* modality; the parts that differ
(how a shard is generated, how the final artifact is assembled, what format it
is recorded as) live behind a single `ModalityHandler` looked up here by
`spec.modality`. This is the *one* dispatch site: adding a modality means
registering a handler, never editing the shared activities or `workflow.py`
(which must stay import-free of modality packages for Temporal determinism).

Handlers are registered at import time by `anodyne_workflows.handlers`, which
`activities` imports once at the bottom of the module. `"tabular"` is the
default -- any modality without a registered handler falls back to it, so the
C0 tabular path is unaffected by this indirection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from anodyne_core.ports import ObjectStore
    from anodyne_dataset.models import DatasetSpec

    from anodyne_workflows.activities import ActivityContext
    from anodyne_workflows.workflow import GenerationInput

_DEFAULT_MODALITY = "tabular"


class ModalityHandler(Protocol):
    """The per-modality behaviour the shared activities dispatch to.

    `shard_rows` sizes `plan_shards`; `artifact_format` is recorded on the
    `DatasetVersion` by `register_version`. `generate_shards`/`assemble` are
    the modality's implementations of the two identically-named activities.
    """

    shard_rows: int
    artifact_format: str

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]: ...

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str: ...


_REGISTRY: dict[str, ModalityHandler] = {}


def register_modality(name: str, handler: ModalityHandler) -> None:
    """Register `handler` for modality `name` (idempotent; last write wins)."""
    _REGISTRY[str(name)] = handler


def get_handler(modality: str | None) -> ModalityHandler:
    """Return the handler for `modality`, falling back to the tabular default."""
    key = str(modality) if modality is not None else _DEFAULT_MODALITY
    handler = _REGISTRY.get(key)
    if handler is not None:
        return handler
    return _REGISTRY[_DEFAULT_MODALITY]


def registered_modalities() -> list[str]:
    """Names of all registered modalities (for diagnostics/tests)."""
    return sorted(_REGISTRY)
