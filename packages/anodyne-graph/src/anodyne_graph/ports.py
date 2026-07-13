"""Ports for the graph modality (adapter-free; the graph analog of `SchemaProposer`).

Only the *ontology proposal* port lives here. Graph generation deliberately
reuses the shape of the existing `Generator` contract — `(spec, start, count,
seed)` — via `anodyne_graph.generator.LLMGraphGenerator`, so the workflow
handler is wired exactly like the other modalities; it is not re-declared as a
new port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from anodyne_core.models import ModelConfig
    from anodyne_core.ports import LLMProvider
    from anodyne_dataset.models import DatasetSpec

    from anodyne_graph.models import GraphOntology


@runtime_checkable
class OntologyProposer(Protocol):
    """Proposes a `GraphOntology` from a dataset spec (its description/directives).

    The analog of `anodyne_dataset.ports.SchemaProposer` for graphs. Stateless:
    the `LLMProvider` and `ModelConfig` are passed per-call rather than held on
    the instance, so a single proposer can serve any tenant/model and later
    waves can compose it freely.
    """

    async def propose(
        self, spec: DatasetSpec, provider: LLMProvider, config: ModelConfig
    ) -> GraphOntology: ...
