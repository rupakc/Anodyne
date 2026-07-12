"""Per-modality dispatch registry for perturbation, a structural mirror of
`anodyne_workflows.modality`.

The `RegistryPerturbator` (the single `Perturbator` adapter) dispatches on
`modality` to a `PerturbationHandler` looked up here -- exactly like the
generation activities dispatch on `spec.modality` via `get_handler`. Adding a
modality means registering a handler, never editing the perturbator. Handlers
self-register at import time in `anodyne_perturbation.handlers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-untyped]
    from anodyne_dataset.models import PerturbationSpec

_DEFAULT_MODALITY = "tabular"


class PerturbationHandler(Protocol):
    """How one modality applies a `PerturbationSpec` to its artifact table."""

    def perturb(self, spec: PerturbationSpec, table: pyarrow.Table, seed: int) -> pyarrow.Table: ...


_REGISTRY: dict[str, PerturbationHandler] = {}


def register_perturbation(name: str, handler: PerturbationHandler) -> None:
    """Register `handler` for modality `name` (idempotent; last write wins)."""
    _REGISTRY[str(name)] = handler


def get_perturbation_handler(modality: str | None) -> PerturbationHandler:
    """Return the handler for `modality`, falling back to the tabular default."""
    key = str(modality) if modality is not None else _DEFAULT_MODALITY
    handler = _REGISTRY.get(key)
    if handler is not None:
        return handler
    return _REGISTRY[_DEFAULT_MODALITY]


def registered_perturbation_modalities() -> list[str]:
    """Names of all registered modalities (for diagnostics/tests)."""
    return sorted(_REGISTRY)
