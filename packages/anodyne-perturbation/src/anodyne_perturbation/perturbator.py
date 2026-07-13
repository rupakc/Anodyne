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
from anodyne_perturbation.registry import (
    get_graph_perturbation_handler,
    get_perturbation_handler,
)

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-untyped]
    from anodyne_dataset.models import PerturbationSpec
    from anodyne_graph.models import GraphDataset


class RegistryPerturbator(Perturbator):
    def perturb(
        self,
        spec: PerturbationSpec,
        table: pyarrow.Table,
        modality: str,
        seed: int,
    ) -> pyarrow.Table:
        return get_perturbation_handler(modality).perturb(spec, table, seed)

    def perturb_graph(
        self,
        spec: PerturbationSpec,
        dataset: GraphDataset,
        seed: int,
        modality: str = "graph",
    ) -> GraphDataset:
        """Route a graph perturbation through the graph modality registry.

        Kept off the `Perturbator` port (whose contract is pa.Table-shaped)
        because graph artifacts are node-link JSON; the perturbation activity
        calls this on the concrete `RegistryPerturbator` for `Modality.GRAPH`.
        """
        return get_graph_perturbation_handler(modality).perturb(spec, dataset, seed)
