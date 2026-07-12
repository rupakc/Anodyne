"""Qualitative expert: the actual LLM-as-a-Judge.

Samples rows/documents, renders them to text, and asks the tenant's model
(strictly through the `LLMProvider` port -- never a vendor SDK) to score realism,
coherence, and task-fit on a 1-5 rubric, returning structured JSON. Works across
modalities: tabular rows are flattened to ``field: value`` lines; a text corpus's
`text_column` documents are sent directly. JSON is fence-stripped and parsed the
same way as `anodyne_generation.proposer.LLMSchemaProposer`.
"""

from __future__ import annotations

import json
import re

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RUBRIC = (
    "You are an expert data reviewer judging the quality of a sample of records. "
    "Rate the sample on three criteria, each an INTEGER from 1 (poor) to 5 (excellent): "
    "realism (do values look like plausible real-world data), coherence (are fields "
    "internally consistent with each other), and task_fit (is the data fit for its stated "
    'purpose). Return ONLY JSON: {"realism": int, "coherence": int, "task_fit": int, '
    '"rationale": str}. No prose outside the JSON.'
)


class QualitativeError(Exception):
    """Raised when the model's output cannot be parsed as a rubric verdict."""


class QualitativeJudge(Judge):
    dimension = EvalDimension.QUALITATIVE

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        sample_text = self._render_sample(ctx)
        req = LLMRequest(
            model_config_id=self._cfg.id,
            messages=[
                Message(role="system", content=_RUBRIC),
                Message(
                    role="user",
                    content=f"Purpose: {ctx.metadata.get('description', 'synthetic dataset')}\n\n"
                    f"Sample records:\n{sample_text}",
                ),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await self._provider.complete(self._cfg, req)
        realism, coherence, task_fit, rationale = self._parse(resp.content)
        score = (realism + coherence + task_fit) / 15.0  # mean of three 1-5 scores, /5
        recs: list[str] = []
        if min(realism, coherence, task_fit) <= 2:
            recs.append(
                "The LLM judge flagged low realism/coherence; inspect the highlighted issues."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=rationale or "LLM rubric verdict.",
            metrics={
                "realism": float(realism),
                "coherence": float(coherence),
                "task_fit": float(task_fit),
            },
            recommendations=recs,
        )

    def _render_sample(self, ctx: EvaluationContext) -> str:
        df = ctx.subject.head(ctx.sample_rows)
        if ctx.text_column and ctx.text_column in df.columns:
            docs = df[ctx.text_column].astype(str).tolist()
            return "\n---\n".join(docs)
        lines: list[str] = []
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            fields = ", ".join(f"{c}: {row[c]}" for c in df.columns)
            lines.append(f"{i}. {fields}")
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> tuple[int, int, int, str]:
        text = raw.strip()
        m = _FENCE.search(text)
        if m:
            text = m.group(1).strip()
        try:
            data = json.loads(text)
            return (
                int(data["realism"]),
                int(data["coherence"]),
                int(data["task_fit"]),
                str(data.get("rationale", "")),
            )
        except Exception as exc:  # json/validation errors -> domain error
            raise QualitativeError(
                f"could not parse rubric verdict from model output: {exc}"
            ) from exc
