"""`LLMGraphGenerator`: ontology + shard + seed -> a `GraphDataset` shard.

Mirrors the `(spec, start, count, seed)` shape of the tabular/text `Generator`
port (it is not a pyarrow generator — a graph is not columnar — so it returns a
`GraphDataset` instead of a `pyarrow.Table`). The workflow `GraphHandler`
serializes the returned dataset to node-link JSON via `serialization`.

Pipeline (the GA "hybrid" skeleton — structure comes from the LLM here; the
richer topology models arrive in wave GB):
  1. One batched, structured LLM call (temperature 0) returns candidate nodes
     and edges keyed by a natural key.
  2. Nodes are validated against the ontology, deduped by (type, natural key),
     assigned stable ids, and their **PII-looking properties are always faked**
     (Faker, seeded) — LLM-produced values for those are never trusted.
  3. Edges are kept only when referentially valid: the edge type exists and its
     endpoints resolve to existing nodes whose types match the edge type's
     declared source/target (domain/range).

Determinism: `np.random.default_rng([seed, shard_index])` + a seeded Faker +
LLM temperature 0. Given the same seed and the same (cached/mocked) LLM
response, the produced graph is identical.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import numpy as np
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec
from faker import Faker

from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.models import (
    Edge,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
    compute_metrics,
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Substrings that mark a property as PII: its value is always faked, never
# taken from the LLM (privacy invariant — see the platform constraints).
_PII_KEYWORDS = ("name", "email", "phone", "address", "ssn", "credit_card", "dob", "birth")


def _is_pii(prop: PropertySpec) -> bool:
    if bool(prop.constraints.get("pii")):
        return True
    lowered = prop.name.lower()
    return any(keyword in lowered for keyword in _PII_KEYWORDS)


def _build_prompt(ontology: GraphOntology, spec: DatasetSpec, count: int) -> list[Message]:
    node_lines = []
    for nt in ontology.node_types:
        props = ", ".join(f"{p.name}:{p.datatype}" for p in nt.properties) or "(no properties)"
        node_lines.append(f"  - {nt.name} {{ {props} }}")
    edge_lines = [
        f"  - {et.name}: {et.source_type} -> {et.target_type}" for et in ontology.edge_types
    ]
    system = (
        "You populate a property graph that conforms to an ontology. Return "
        "ONLY a JSON object:\n"
        '{"nodes": [{"type": <node type>, "key": <short unique natural key>, '
        '"properties": {<prop name>: <value>}}], '
        '"edges": [{"type": <edge type>, "source": <a node key>, '
        '"target": <a node key>, "properties": {}}]}\n'
        "Node types:\n" + "\n".join(node_lines) + "\n"
        "Edge types:\n" + ("\n".join(edge_lines) or "  (none)") + "\n"
        "Every edge's source/target MUST be the key of a node whose type "
        "matches the edge type's declared source/target. No prose."
    )
    user = (
        f"Domain: {spec.description or spec.name}\n"
        f"Generate about {count} nodes total across the node types, plus "
        "plausible edges between them. Return only the JSON object."
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _extract_object(content: str) -> dict[str, Any]:
    raw = content.strip()
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GraphGenerationError(f"could not parse graph JSON from model output: {exc}") from exc
    if not isinstance(data, dict):
        raise GraphGenerationError("graph output was valid JSON but not an object")
    return data


def _coerce(value: Any, datatype: str, rng: np.random.Generator) -> Any:
    """Best-effort coercion of an LLM-provided value to the declared datatype."""
    try:
        if datatype == "integer":
            return int(value)
        if datatype == "float":
            return float(value)
        if datatype == "boolean":
            return bool(value) if not isinstance(value, str) else value.strip().lower() == "true"
    except (TypeError, ValueError):
        pass
    return value


class LLMGraphGenerator:
    """LLM-backed graph generator for one shard. Constructed with the tenant's
    `LLMProvider` + `ModelConfig`, mirroring `anodyne_text.generator.TextGenerator`."""

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    def generate(
        self,
        spec: DatasetSpec,
        start_index: int,
        count: int,
        seed: int,
        shard_index: int = 0,
    ) -> GraphDataset:
        """Generate one graph shard as a `GraphDataset`.

        `start_index`/`count` bound the shard's node budget (GA uses a single
        shard covering all nodes; the range is the seam for wave-GB
        partitioning). `shard_index` diversifies the RNG per shard.

        Raises:
            GraphGenerationError: if no ontology is present on the spec or no
                valid nodes can be produced.
        """
        ontology = self._ontology(spec)
        rng = np.random.default_rng([seed, shard_index])
        fake = Faker()
        Faker.seed(seed * 1_000_003 + shard_index * 7919 + start_index)

        request = LLMRequest(
            model_config_id=self._cfg.id,
            messages=_build_prompt(ontology, spec, count),
            params={"temperature": 0, "seed": seed},
        )
        response = asyncio.run(self._provider.complete(self._cfg, request))
        data = _extract_object(response.content)

        nodes, key_to_id = self._build_nodes(ontology, data.get("nodes"), count, rng, fake)
        if not nodes:
            raise GraphGenerationError(
                f"no valid nodes produced for dataset {spec.id} (shard {shard_index})"
            )
        edges = self._build_edges(ontology, data.get("edges"), nodes, key_to_id, rng)
        return GraphDataset(
            ontology=ontology,
            nodes=nodes,
            edges=edges,
            metrics=compute_metrics(nodes, edges),
        )

    def _ontology(self, spec: DatasetSpec) -> GraphOntology:
        raw = spec.directives.get("ontology")
        if raw is None:
            raise GraphGenerationError(
                f"dataset {spec.id} has no ontology in directives['ontology']; "
                "propose or set one before generating a graph"
            )
        if isinstance(raw, GraphOntology):
            return raw
        return GraphOntology.model_validate(raw)

    def _build_nodes(
        self,
        ontology: GraphOntology,
        raw_nodes: Any,
        count: int,
        rng: np.random.Generator,
        fake: Faker,
    ) -> tuple[list[Node], dict[tuple[str, str], str]]:
        nodes: list[Node] = []
        # (type, natural key) -> assigned node id; also the dedup index.
        key_to_id: dict[tuple[str, str], str] = {}
        if not isinstance(raw_nodes, list):
            return nodes, key_to_id
        for item in raw_nodes:
            if len(nodes) >= count:
                break
            if not isinstance(item, dict):
                continue
            ntype = item.get("type")
            node_type = ontology.node_type(ntype) if isinstance(ntype, str) else None
            if node_type is None:
                continue  # unknown type: not in the ontology
            key = item.get("key")
            key = str(key) if key is not None else str(len(nodes))
            dedup_key = (node_type.name, key)
            if dedup_key in key_to_id:
                continue  # dedup by (type, natural key)
            node_id = f"{node_type.name}:{key}"
            props = self._node_properties(node_type, item.get("properties"), rng, fake)
            nodes.append(Node(id=node_id, type=node_type.name, properties=props))
            key_to_id[dedup_key] = node_id
        return nodes, key_to_id

    def _node_properties(
        self,
        node_type: NodeType,
        raw_props: Any,
        rng: np.random.Generator,
        fake: Faker,
    ) -> dict[str, Any]:
        given = raw_props if isinstance(raw_props, dict) else {}
        props: dict[str, Any] = {}
        for prop in node_type.properties:
            if _is_pii(prop):
                # Always fake PII — never trust the LLM value.
                props[prop.name] = self._fake_pii(prop, fake)
            elif prop.name in given and given[prop.name] is not None:
                props[prop.name] = _coerce(given[prop.name], prop.datatype, rng)
            else:
                props[prop.name] = self._synthesize(prop, rng, fake)
        return props

    @staticmethod
    def _fake_pii(prop: PropertySpec, fake: Faker) -> Any:
        lowered = prop.name.lower()
        if "email" in lowered:
            return fake.email()
        if "phone" in lowered:
            return fake.phone_number()
        if "address" in lowered:
            return fake.address().replace("\n", ", ")
        if "name" in lowered:
            return fake.name()
        if "ssn" in lowered:
            return fake.ssn()
        if "dob" in lowered or "birth" in lowered:
            return fake.date_of_birth().isoformat()
        return fake.word()

    @staticmethod
    def _synthesize(prop: PropertySpec, rng: np.random.Generator, fake: Faker) -> Any:
        choices = prop.constraints.get("choices")
        if isinstance(choices, list) and choices:
            return choices[int(rng.integers(0, len(choices)))]
        if prop.datatype == "integer":
            return int(rng.integers(0, 1000))
        if prop.datatype == "float":
            return float(rng.random())
        if prop.datatype == "boolean":
            return bool(rng.random() < 0.5)
        if prop.datatype == "datetime":
            return fake.date_time().isoformat()
        return fake.word()

    def _build_edges(
        self,
        ontology: GraphOntology,
        raw_edges: Any,
        nodes: list[Node],
        key_to_id: dict[tuple[str, str], str],
        rng: np.random.Generator,
    ) -> list[Edge]:
        edges: list[Edge] = []
        if not isinstance(raw_edges, list):
            return edges
        node_type_by_id = {n.id: n.type for n in nodes}
        # LLM edges reference nodes by natural key (per the prompt); resolve each
        # endpoint to a concrete node id of the edge's expected endpoint type.
        seen: set[tuple[str, str, str]] = set()
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            etype = item.get("type")
            edge_type = ontology.edge_type(etype) if isinstance(etype, str) else None
            if edge_type is None:
                continue
            src_id = self._resolve(item.get("source"), edge_type.source_type, key_to_id)
            tgt_id = self._resolve(item.get("target"), edge_type.target_type, key_to_id)
            if src_id is None or tgt_id is None:
                continue  # dangling reference: drop for referential integrity
            # Belt-and-braces domain/range check against resolved node types.
            if (
                node_type_by_id.get(src_id) != edge_type.source_type
                or node_type_by_id.get(tgt_id) != edge_type.target_type
            ):
                continue
            signature = (edge_type.name, src_id, tgt_id)
            if signature in seen:
                continue
            seen.add(signature)
            raw_props = item.get("properties")
            props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
            edges.append(
                Edge(
                    id=f"{edge_type.name}:{len(edges)}",
                    type=edge_type.name,
                    source=src_id,
                    target=tgt_id,
                    properties=props,
                )
            )
        return edges

    @staticmethod
    def _resolve(key: Any, expected_type: str, key_to_id: dict[tuple[str, str], str]) -> str | None:
        if key is None:
            return None
        return key_to_id.get((expected_type, str(key)))
