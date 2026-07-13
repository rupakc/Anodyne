"""GraphRAG fixture serialization: JSONL items + a describing manifest.

``fixture_to_jsonl`` emits one JSON object per QA item (newline-delimited), a
GraphRAG-eval-friendly shape carrying the question, the graph-derived gold
answer, the grounding node ids, and the gold supporting path. Objects use
``sort_keys`` so the same fixture serializes byte-identically (determinism /
checksum contract).
"""

from __future__ import annotations

import json
from typing import Any

from anodyne_graph.graphrag.models import GraphQAFixture, GraphQAItem


def _item_to_obj(item: GraphQAItem) -> dict[str, Any]:
    return {
        "question": item.question,
        "answer": item.answer,
        "answer_node_ids": item.answer_node_ids,
        "hop_count": item.hop_count,
        "question_type": item.question_type,
        "difficulty": item.difficulty,
        "gold_path": {
            "hops": [list(hop) for hop in item.gold_path.hops],
            "node_ids": item.gold_path.node_ids,
            "edge_ids": item.gold_path.edge_ids,
            "terminal_node_id": item.gold_path.terminal_node_id,
        },
    }


def fixture_to_jsonl(fixture: GraphQAFixture) -> bytes:
    """Serialize the fixture to newline-delimited JSON (one object per item)."""
    lines = [
        json.dumps(_item_to_obj(item), sort_keys=True, ensure_ascii=False) for item in fixture.items
    ]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    return text.encode("utf-8")


def graphrag_manifest(fixture: GraphQAFixture) -> dict[str, Any]:
    """A compact manifest describing the fixture (counts + distributions)."""
    type_counts: dict[str, int] = {}
    hop_counts: dict[str, int] = {}
    for item in fixture.items:
        type_counts[item.question_type] = type_counts.get(item.question_type, 0) + 1
        key = str(item.hop_count)
        hop_counts[key] = hop_counts.get(key, 0) + 1
    return {
        "dataset_version_id": fixture.dataset_version_id,
        "item_count": len(fixture.items),
        "seed": fixture.metadata.get("seed"),
        "question_type_counts": dict(sorted(type_counts.items())),
        "hop_count_distribution": dict(sorted(hop_counts.items())),
        "metadata": fixture.metadata,
    }
