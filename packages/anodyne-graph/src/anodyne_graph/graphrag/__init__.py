"""GraphRAG multi-hop QA fixtures (Wave 2, track GG).

From a generated ``GraphDataset`` this synthesizes a GraphRAG evaluation
fixture: multi-hop questions, each grounded in a concrete path through the
graph, with a gold answer and the gold supporting path.

**Groundedness invariant (the whole point):** every answer is derived from the
graph traversal, never from the LLM. The LLM (via the ``LLMProvider`` port) only
rewrites the question's *surface phrasing*; if its output is empty the template
phrasing is used verbatim. Path sampling and template selection are seeded and
deterministic, so an identical fixture is produced across two runs with the same
seed and the same (fake or real) provider.
"""

from anodyne_graph.graphrag.export import fixture_to_jsonl, graphrag_manifest
from anodyne_graph.graphrag.models import GraphQAFixture, GraphQAItem, QAPath
from anodyne_graph.graphrag.pathfinder import sample_paths
from anodyne_graph.graphrag.question_gen import GraphRAGGenerator

__all__ = [
    "GraphQAFixture",
    "GraphQAItem",
    "GraphRAGGenerator",
    "QAPath",
    "fixture_to_jsonl",
    "graphrag_manifest",
    "sample_paths",
]
