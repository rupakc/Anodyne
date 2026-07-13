"""Hybrid engine: procedural structure + LLM-filled semantics.

The topology (which nodes, which typed edges) is built deterministically by the
``ProceduralTopologyGenerator``; the LLM then fills *non-PII* node attributes
with realistic, domain-appropriate values in one batched, structured call
(temperature 0). PII-looking properties are **never** sent to or taken from the
LLM -- they stay faked by the topology step (privacy invariant).

Determinism honesty (matches the text modality / design spec): the topology is
bit-exact reproducible; the LLM fill is best-effort reproducible (temp 0, seeded
prompt, and identical given a cached/mocked response). We do not over-promise
bit-exact whole graphs when a live LLM is in the loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_graph.models import GraphDataset, GraphOntology, Node
from anodyne_graph.properties import coerce, is_pii
from anodyne_graph.topology import ProceduralTopologyGenerator

if TYPE_CHECKING:
    from anodyne_dataset.models import DatasetSpec

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _fillable_props(ontology: GraphOntology) -> dict[str, list[str]]:
    """Non-PII, non-enum property names per node type the LLM may fill."""
    out: dict[str, list[str]] = {}
    for nt in ontology.node_types:
        names = [
            p.name for p in nt.properties if not is_pii(p) and not p.constraints.get("choices")
        ]
        if names:
            out[nt.name] = names
    return out


def _build_prompt(
    ontology: GraphOntology,
    fillable: dict[str, list[str]],
    spec: DatasetSpec,
    counts: dict[str, int],
) -> list[Message]:
    lines = []
    for ntype, props in fillable.items():
        dtypes = {
            p.name: p.datatype
            for nt in ontology.node_types
            if nt.name == ntype
            for p in nt.properties
        }
        cols = ", ".join(f"{p}:{dtypes.get(p, 'string')}" for p in props)
        lines.append(f'  "{ntype}": up to {counts.get(ntype, 0)} objects of {{ {cols} }}')
    system = (
        "You fill realistic attribute values for entities in a property graph. "
        "Return ONLY a JSON object mapping each node type to a list of objects; "
        "each object gives values for that type's listed properties. Do NOT "
        "include names or other personal identifiers (they are generated "
        "separately). Shape:\n{\n" + ",\n".join(lines) + "\n}\nNo prose."
    )
    user = f"Domain: {spec.description or spec.name}\nReturn only the JSON object."
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _extract_object(content: str) -> dict[str, Any]:
    raw = content.strip()
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class HybridGraphGenerator:
    """Procedural topology + LLM semantic fill. Same ``generate`` shape as GA."""

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config
        self._topology = ProceduralTopologyGenerator()

    def generate(
        self,
        spec: DatasetSpec,
        start_index: int,
        count: int,
        seed: int,
        shard_index: int = 0,
    ) -> GraphDataset:
        base = self._topology.generate(spec, start_index, count, seed, shard_index)
        fillable = _fillable_props(base.ontology)
        if fillable:
            self._fill(base, fillable, spec, seed)
        base.metrics["engine"] = "hybrid"
        return base

    def _fill(
        self,
        dataset: GraphDataset,
        fillable: dict[str, list[str]],
        spec: DatasetSpec,
        seed: int,
    ) -> None:
        counts: dict[str, int] = {}
        for node in dataset.nodes:
            counts[node.type] = counts.get(node.type, 0) + 1
        request = LLMRequest(
            model_config_id=self._cfg.id,
            messages=_build_prompt(dataset.ontology, fillable, spec, counts),
            params={"temperature": 0, "seed": seed},
        )
        response = asyncio.run(self._provider.complete(self._cfg, request))
        values = _extract_object(response.content)

        dtype_of = {
            (nt.name, p.name): p.datatype
            for nt in dataset.ontology.node_types
            for p in nt.properties
        }
        # Per node type, consume the LLM's value list in node order (cycling if
        # the model returned fewer rows than nodes); non-PII props only.
        cursor: dict[str, int] = {}
        for node in dataset.nodes:
            allowed = fillable.get(node.type)
            if not allowed:
                continue
            rows = values.get(node.type)
            if not isinstance(rows, list) or not rows:
                continue
            idx = cursor.get(node.type, 0)
            row = rows[idx % len(rows)]
            cursor[node.type] = idx + 1
            if not isinstance(row, dict):
                continue
            self._apply_row(node, row, allowed, dtype_of)

    @staticmethod
    def _apply_row(
        node: Node,
        row: dict[str, Any],
        allowed: list[str],
        dtype_of: dict[tuple[str, str], str],
    ) -> None:
        for prop in allowed:
            if prop in row and row[prop] is not None:
                node.properties[prop] = coerce(row[prop], dtype_of.get((node.type, prop), "string"))
