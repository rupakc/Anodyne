"""Engine selection + constraint post-processing for the graph modality.

One place decides which generation engine a `DatasetSpec` selects, so the
workflow ``GraphHandler`` stays a thin caller and the GA path is unchanged when
no directive is set. Selection (first match wins):

- ``source == "sample"`` (or ``directives["method"] == "from_sample"``)
  -> ``FromSampleGraphGenerator`` (needs a parsed sample graph);
- ``directives["method"] == "hybrid"`` -> ``HybridGraphGenerator``;
- ``directives["topology"]`` present -> ``ProceduralTopologyGenerator``;
- otherwise -> GA's ``LLMGraphGenerator`` (the default -- GA behaviour intact).

``generate_shard`` runs the selected engine, then applies the ontology
constraint layer per directives (``inject_violations``, ``validate`` /
``validate_shacl``, ``cardinality``), recording results in the dataset metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec

from anodyne_graph.constraints import OntologyConstraintValidator, inject_violations
from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.from_sample import FromSampleGraphGenerator
from anodyne_graph.generator import LLMGraphGenerator
from anodyne_graph.hybrid import HybridGraphGenerator
from anodyne_graph.models import GraphDataset
from anodyne_graph.topology import ProceduralTopologyGenerator

if TYPE_CHECKING:
    pass


class GraphEngine(Protocol):
    """The shared engine shape (mirrors the platform `Generator` contract)."""

    def generate(
        self, spec: DatasetSpec, start_index: int, count: int, seed: int, shard_index: int = 0
    ) -> GraphDataset: ...


def _method(spec: DatasetSpec) -> str:
    return str(spec.directives.get("method", "")).lower()


def is_from_sample(spec: DatasetSpec) -> bool:
    return spec.source == "sample" or _method(spec) == "from_sample"


def needs_llm(spec: DatasetSpec) -> bool:
    """True when the selected engine drives the LLM (default or hybrid)."""
    if is_from_sample(spec):
        return False
    if _method(spec) == "hybrid":
        return True
    return not spec.directives.get("topology")


def build_graph_engine(
    spec: DatasetSpec,
    provider: LLMProvider | None,
    model_config: ModelConfig | None,
    *,
    sample: GraphDataset | None = None,
) -> GraphEngine:
    """Select the generation engine for ``spec`` (see module docstring)."""
    if is_from_sample(spec):
        if sample is None:
            raise GraphGenerationError(
                "from-sample graph generation requires an uploaded sample graph"
            )
        return FromSampleGraphGenerator(sample)
    if _method(spec) == "hybrid":
        return HybridGraphGenerator(*_require_llm_pair(provider, model_config))
    if spec.directives.get("topology"):
        return ProceduralTopologyGenerator()
    return LLMGraphGenerator(*_require_llm_pair(provider, model_config))


def _require_llm_pair(
    provider: LLMProvider | None, model_config: ModelConfig | None
) -> tuple[LLMProvider, ModelConfig]:
    if provider is None or model_config is None:
        raise GraphGenerationError("this engine requires an LLM provider + model config")
    return provider, model_config


def apply_constraints(
    dataset: GraphDataset, spec: DatasetSpec, seed: int, shard_index: int
) -> GraphDataset:
    """Apply the ontology constraint layer per directives; record in metrics."""
    directives = spec.directives
    raw_inject = directives.get("inject_violations", 0) or 0
    inject = int(raw_inject) if isinstance(raw_inject, (int, float, str)) else 0
    validator = OntologyConstraintValidator()
    if inject > 0:
        rng = np.random.default_rng([seed, shard_index, 99])
        dataset = inject_violations(dataset, inject, rng)
    want_check = (
        bool(directives.get("validate")) or bool(directives.get("validate_shacl")) or inject > 0
    )
    if want_check:
        cardinality = directives.get("cardinality")
        report = validator.check(dataset, cardinality if isinstance(cardinality, dict) else None)
        dataset.metrics["constraint_conforms"] = report.conforms
        dataset.metrics["constraint_violations"] = report.count
    if directives.get("validate_shacl"):
        shacl = validator.validate_shacl(dataset)
        dataset.metrics["shacl_conforms"] = shacl.conforms
        dataset.metrics["shacl_violations"] = shacl.count
    return dataset


def generate_shard(
    spec: DatasetSpec,
    provider: LLMProvider | None,
    model_config: ModelConfig | None,
    start_index: int,
    count: int,
    seed: int,
    shard_index: int = 0,
    *,
    sample: GraphDataset | None = None,
) -> GraphDataset:
    """Select an engine, generate one shard, apply the constraint layer."""
    engine = build_graph_engine(spec, provider, model_config, sample=sample)
    dataset = engine.generate(spec, start_index, count, seed, shard_index)
    return apply_constraints(dataset, spec, seed, shard_index)
