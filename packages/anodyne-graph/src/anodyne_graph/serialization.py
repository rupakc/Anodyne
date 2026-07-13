"""Canonical graph artifact format: **node-link JSON**.

This is the on-disk / on-wire representation of a generated graph — the single
artifact the generation workflow uploads and the download route streams back.
Later waves (export, evaluation, UI) read graphs *through this schema*, so it is
the frozen interchange contract for the graph modality.

Schema (a single JSON object, UTF-8):

    {
      "ontology": {                     # the T-Box (GraphOntology)
        "node_types": [ {"name": str, "properties": [PropertySpec...]} ],
        "edge_types": [ {"name": str, "source_type": str, "target_type": str,
                          "properties": [PropertySpec...], "directed": bool} ],
        "subclass_of": { childType: parentType }
      },
      "nodes": [ {"id": str, "type": str, "properties": {..}} ],
      "edges": [ {"id": str, "type": str, "source": str, "target": str,
                   "properties": {..}} ],
      "metrics": { "node_count": int, "edge_count": int,
                    "nodes_by_type": {type: int}, "edges_by_type": {type: int} }
    }

`source`/`target` on an edge are `Node.id` references. The format is the
node-link convention (a `nodes` list + a `links`/`edges` list) used by common
graph tooling, so a downstream exporter can map it to GraphML/Cypher/RDF etc.
without re-deriving structure.

`to_json_bytes` / `from_json_bytes` are exact round-trip inverses.
"""

from __future__ import annotations

import json

from anodyne_graph.models import GraphDataset


def to_json_bytes(dataset: GraphDataset) -> bytes:
    """Serialize a `GraphDataset` to canonical node-link JSON bytes.

    Uses `sort_keys` so the same dataset always serializes to byte-identical
    output (supports the determinism + checksum contracts).
    """
    payload = dataset.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def from_json_bytes(data: bytes) -> GraphDataset:
    """Parse canonical node-link JSON bytes back into a `GraphDataset`.

    Raises `ValueError` on malformed input (invalid JSON or a payload that does
    not satisfy the `GraphDataset` schema).
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"not valid graph node-link JSON: {exc}") from exc
    return GraphDataset.model_validate(payload)
