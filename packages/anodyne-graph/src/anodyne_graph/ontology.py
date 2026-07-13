"""`LLMOntologyProposer`: description -> `GraphOntology` via the `LLMProvider` port.

Mirrors `anodyne_generation.proposer.LLMSchemaProposer` (same fenced-JSON
extraction + validate/repair-or-raise stance), but produces a property-graph
ontology (node types + edge types) instead of a flat tabular schema.
"""

from __future__ import annotations

import json
import re
from typing import Any

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec

from anodyne_graph.errors import OntologyProposalError
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_SYSTEM = (
    "You design property-graph ontologies (knowledge-graph schemas). Given a "
    "domain description, return ONLY a JSON object with this exact shape:\n"
    '{"node_types": [{"name": str, "properties": [{"name": str, "datatype": '
    'one of ["string","integer","float","boolean","datetime"], "nullable": '
    'bool (optional)}]}], "edge_types": [{"name": str, "source_type": str '
    '(a node_type name), "target_type": str (a node_type name), '
    '"directed": bool (optional), "properties": [ ... same as above ... ]}]}\n'
    "Node/edge type names should be PascalCase/UPPER_SNAKE respectively. Every "
    "edge_type's source_type and target_type MUST be a declared node_type. No prose."
)


def _extract_json_object(content: str) -> dict[str, Any]:
    raw = content.strip()
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OntologyProposalError(f"could not parse ontology JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OntologyProposalError("ontology output was valid JSON but not an object")
    return data


def _property(item: Any) -> PropertySpec | None:
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        return None
    datatype = item.get("datatype")
    raw_constraints = item.get("constraints")
    constraints: dict[str, Any] = raw_constraints if isinstance(raw_constraints, dict) else {}
    return PropertySpec(
        name=item["name"],
        datatype=datatype if isinstance(datatype, str) and datatype else "string",
        nullable=bool(item.get("nullable", False)),
        constraints=constraints,
    )


def _properties(raw: Any) -> list[PropertySpec]:
    if not isinstance(raw, list):
        return []
    return [p for p in (_property(i) for i in raw) if p is not None]


class LLMOntologyProposer:
    """LLM-backed `OntologyProposer` (satisfies the `ports.OntologyProposer` Protocol).

    Stateless: `provider`/`config` are supplied per `propose` call. The LLM is
    asked at temperature 0 for reproducibility, matching the platform stance.
    """

    async def propose(
        self, spec: DatasetSpec, provider: LLMProvider, config: ModelConfig
    ) -> GraphOntology:
        """Propose an ontology from the spec's description (+ any directives).

        Raises:
            OntologyProposalError: if the LLM output cannot be parsed/repaired
                into an ontology with at least one node type.
        """
        directives = "\n".join(f"- {k}: {v}" for k, v in spec.directives.items())
        user = spec.description or spec.name
        if directives:
            user = f"{user}\n\nAdditional directives:\n{directives}"
        request = LLMRequest(
            model_config_id=config.id,
            messages=[
                Message(role="system", content=_SYSTEM),
                Message(role="user", content=user),
            ],
            params={"temperature": 0},
        )
        response = await provider.complete(config, request)
        data = _extract_json_object(response.content)

        node_types: list[NodeType] = []
        for item in data.get("node_types", []) or []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                node_types.append(
                    NodeType(name=item["name"], properties=_properties(item.get("properties")))
                )
        if not node_types:
            raise OntologyProposalError(
                "proposed ontology has no node types; provide a more specific description"
            )

        names = {nt.name for nt in node_types}
        edge_types: list[EdgeType] = []
        for item in data.get("edge_types", []) or []:
            if not isinstance(item, dict):
                continue
            name, src, tgt = item.get("name"), item.get("source_type"), item.get("target_type")
            # Drop edges whose endpoints aren't declared node types (referential
            # integrity of the schema itself) rather than emitting a broken ontology.
            if not (isinstance(name, str) and src in names and tgt in names):
                continue
            edge_types.append(
                EdgeType(
                    name=name,
                    source_type=src,
                    target_type=tgt,
                    directed=bool(item.get("directed", True)),
                    properties=_properties(item.get("properties")),
                )
            )
        return GraphOntology(node_types=node_types, edge_types=edge_types)
