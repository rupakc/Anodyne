"""`RegistryPerturbator`: the single `Perturbator` adapter.

Dispatches on `modality` through the perturbation modality registry -- the one
dispatch site, mirroring generation's `get_handler`. Importing this module
imports `handlers` for its registration side effect, so the registry is
populated by the time anyone calls `perturb`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anodyne_dataset.ports import Perturbator

import anodyne_perturbation.handlers  # noqa: F401  (registration side effect)
from anodyne_perturbation.registry import get_perturbation_handler

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-untyped]
    from anodyne_dataset.models import PerturbationSpec


class RegistryPerturbator(Perturbator):
    def perturb(
        self,
        spec: PerturbationSpec,
        table: pyarrow.Table,
        modality: str,
        seed: int,
    ) -> pyarrow.Table:
        return get_perturbation_handler(modality).perturb(spec, table, seed)
