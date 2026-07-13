"""`PerturbationHandler` implementations + their registration.

Imported once (for its registration side effects) from
`anodyne_perturbation.perturbator` and by the workflow activities. Tabular and
text are fully implemented; image/audio/video are registered as a clean
`_UnsupportedModalityHandler` seam -- a real media perturbator would replace
these (no fake media logic is stubbed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anodyne_perturbation.registry import register_graph_perturbation, register_perturbation
from anodyne_perturbation.tabular import perturb_tabular
from anodyne_perturbation.text import perturb_text

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-untyped]
    from anodyne_dataset.models import PerturbationSpec
    from anodyne_graph.models import GraphDataset


class TabularPerturbationHandler:
    def perturb(self, spec: PerturbationSpec, table: pyarrow.Table, seed: int) -> pyarrow.Table:
        return perturb_tabular(spec, table, seed)


class TextPerturbationHandler:
    def perturb(self, spec: PerturbationSpec, table: pyarrow.Table, seed: int) -> pyarrow.Table:
        return perturb_text(spec, table, seed)


class _UnsupportedModalityHandler:
    """Registered seam for a modality whose perturbation isn't implemented yet.

    Keeps dispatch uniform (every modality resolves to a handler) while making
    the gap explicit rather than silently corrupting binary media.
    """

    def __init__(self, modality: str) -> None:
        self._modality = modality

    def perturb(self, spec: PerturbationSpec, table: pyarrow.Table, seed: int) -> pyarrow.Table:
        raise NotImplementedError(
            f"perturbation is not yet implemented for modality {self._modality!r}; "
            "register a real handler via anodyne_perturbation.register_perturbation"
        )


class GraphPerturbationHandler:
    """Routes graph perturbations to `anodyne_graph.perturb.perturb_graph`.

    Lives here (not in `anodyne-graph`) so the graph perturbation *dispatch*
    follows the same self-registration pattern as the columnar handlers while
    the corruption math stays in the graph package. Unlike the columnar
    handlers it consumes/produces a `GraphDataset`, not a `pyarrow.Table` -- a
    graph artifact is node-link JSON and must never be forced through pa.Table.
    """

    def perturb(self, spec: PerturbationSpec, dataset: GraphDataset, seed: int) -> GraphDataset:
        from anodyne_graph.perturb import perturb_graph

        return perturb_graph(dataset, spec.family, spec.intensity, seed, dict(spec.params))


register_perturbation("tabular", TabularPerturbationHandler())
register_perturbation("text", TextPerturbationHandler())
register_perturbation("image", _UnsupportedModalityHandler("image"))
register_perturbation("audio", _UnsupportedModalityHandler("audio"))
register_perturbation("video", _UnsupportedModalityHandler("video"))
register_graph_perturbation("graph", GraphPerturbationHandler())
