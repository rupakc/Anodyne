"""Semantic-plausibility expert: the LLM-as-a-Judge for graphs.

Deterministically samples triples (edges rendered as
``(source_label:SourceType) -[relation]-> (target_label:TargetType)`` with a few
salient properties) and asks the tenant's model — strictly through the
`LLMProvider` port, never a vendor SDK — to rate the sample on a 1-5 rubric for
**realism** (do the entities/relations look like real-world knowledge) and
**coherence** (are the relations logically consistent with the entity types).
The verdict is parsed from fenced JSON exactly like `QualitativeJudge`. The LLM
is mocked in unit tests.
"""

from __future__ import annotations

import json
import re

import numpy as np
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.graph_judges.base import node_label
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RUBRIC = (
    "You are a knowledge-graph reviewer judging a sample of triples from a synthetic "
    "graph. Rate the sample on two criteria, each an INTEGER from 1 (poor) to 5 "
    "(excellent): realism (do the entities and relations resemble real-world knowledge) "
    "and coherence (are the relations logically consistent with the entity types). "
    'Return ONLY JSON: {"realism": int, "coherence": int, "rationale": str}. '
    "No prose outside the JSON."
)


class SemanticPlausibilityError(Exception):
    """Raised when the model's output cannot be parsed as a rubric verdict."""


class SemanticPlausibilityGraphJudge(Judge):
    dimension = EvalDimension.GRAPH_SEMANTIC

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        if ctx.subject_graph is None:
            raise JudgeNotApplicable("semantic plausibility requires a graph artifact")
        graph = ctx.subject_graph
        if not graph.edges:
            raise JudgeNotApplicable("semantic plausibility requires at least one edge")

        sample_text = self._render_triples(ctx)
        domain = ctx.metadata.get("description", "synthetic knowledge graph")
        req = LLMRequest(
            model_config_id=self._cfg.id,
            messages=[
                Message(role="system", content=_RUBRIC),
                Message(
                    role="user",
                    content=f"Domain: {domain}\n\nSampled triples:\n{sample_text}",
                ),
            ],
            params={"temperature": 0},
        )
        resp = await self._provider.complete(self._cfg, req)
        realism, coherence, rationale = self._parse(resp.content)
        score = (realism + coherence) / 10.0
        recs: list[str] = []
        if min(realism, coherence) <= 2:
            recs.append(
                "The LLM judge flagged implausible triples; inspect entity/relation labels."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=rationale or "LLM triple-plausibility verdict.",
            metrics={"realism": float(realism), "coherence": float(coherence)},
            recommendations=recs,
        )

    def _render_triples(self, ctx: EvaluationContext) -> str:
        graph = ctx.subject_graph
        assert graph is not None  # guarded by evaluate
        by_id = {n.id: n for n in graph.nodes}
        rng = np.random.default_rng(ctx.seed)
        k = min(ctx.sample_rows, len(graph.edges))
        idx = rng.choice(len(graph.edges), size=k, replace=False)
        lines: list[str] = []
        for i in sorted(int(j) for j in idx):
            edge = graph.edges[i]
            src, tgt = by_id.get(edge.source), by_id.get(edge.target)
            if src is None or tgt is None:
                continue
            src_str = f"{node_label(src)}:{src.type}"
            tgt_str = f"{node_label(tgt)}:{tgt.type}"
            lines.append(f"({src_str}) -[{edge.type}]-> ({tgt_str})")
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> tuple[int, int, str]:
        text = raw.strip()
        m = _FENCE.search(text)
        if m:
            text = m.group(1).strip()
        try:
            data = json.loads(text)
            return (
                int(data["realism"]),
                int(data["coherence"]),
                str(data.get("rationale", "")),
            )
        except Exception as exc:
            raise SemanticPlausibilityError(
                f"could not parse rubric verdict from model output: {exc}"
            ) from exc
