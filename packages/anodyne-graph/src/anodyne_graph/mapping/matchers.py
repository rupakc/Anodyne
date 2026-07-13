"""Matchers for the hybrid ontology aligner: lexical, embedding, LLM.

All three are deterministic given their inputs (the LLM at temperature 0 and a
fixed fake in tests; the embedding matcher given fixed embeddings). No live
service is contacted here — the LLM is reached only through the `LLMProvider`
port and embeddings through an injected `EmbeddingFn`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider
from pydantic import BaseModel

from anodyne_graph.mapping.models import MappingRelation
from anodyne_graph.mapping.ports import EmbeddingFn

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def _tokens(label: str) -> list[str]:
    """Normalize an entity label to lowercase word tokens.

    Splits camelCase boundaries and any non-alphanumeric separators
    (snake/kebab/space), so "WorksAt", "works_at" and "WORKS AT" all yield
    ["works", "at"]. Deterministic.
    """
    spaced = _CAMEL.sub(" ", label)
    spaced = _NON_ALNUM.sub(" ", spaced)
    return [t for t in spaced.lower().split() if t]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


class LexicalMatcher:
    """Deterministic label similarity: token-Jaccard + normalized edit distance.

    `score` returns 1.0 for labels that normalize identically; otherwise the
    mean of token-set Jaccard and (1 - normalized Levenshtein) over the
    normalized strings. Stdlib + numpy only, no external services.
    """

    name = "lexical"
    available = True

    def __init__(self, *, jaccard_weight: float = 0.5) -> None:
        self._jw = jaccard_weight

    def score(self, source: str, target: str) -> float:
        ta, tb = _tokens(source), _tokens(target)
        na, nb = " ".join(ta), " ".join(tb)
        if na and na == nb:
            return 1.0
        sa, sb = set(ta), set(tb)
        union = sa | sb
        jaccard = len(sa & sb) / len(union) if union else 0.0
        max_len = max(len(na), len(nb))
        edit_sim = 1.0 - (_levenshtein(na, nb) / max_len) if max_len else 0.0
        return self._jw * jaccard + (1.0 - self._jw) * edit_sim


class EmbeddingMatcher:
    """Cosine similarity of injected embeddings, mapped to [0, 1].

    Takes an `EmbeddingFn` (the `LLMProvider` port exposes no `embed`, so the
    embedding function is injected). If none is supplied the matcher reports
    `available = False` and `score` returns 0.0 so the aligner degrades to
    lexical-only. Embeddings are cached per label for determinism and to avoid
    recomputation.
    """

    name = "embedding"

    def __init__(self, embed_fn: EmbeddingFn | None) -> None:
        self._embed = embed_fn
        self.available = embed_fn is not None
        self._cache: dict[str, list[float]] = {}

    def _vector(self, label: str) -> list[float]:
        if label not in self._cache:
            assert self._embed is not None
            self._cache[label] = [float(x) for x in self._embed([label])[0]]
        return self._cache[label]

    def score(self, source: str, target: str) -> float:
        if self._embed is None:
            return 0.0
        va, vb = self._vector(source), self._vector(target)
        dot = sum(x * y for x, y in zip(va, vb, strict=False))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        cos = max(-1.0, min(1.0, dot / (na * nb)))
        return (cos + 1.0) / 2.0


class LLMJudgement(BaseModel):
    """The LLM's adjudication of a candidate pair."""

    predicate: MappingRelation
    confidence: float
    justification: str


_LLM_SYSTEM = (
    "You align entities across two knowledge-graph ontologies. Given a source "
    "and target entity (name + description), decide the SKOS mapping relation "
    "and how confident you are. Return ONLY a JSON object:\n"
    '{"predicate": one of ["exact_match","close_match","broad_match",'
    '"narrow_match","related_match"], "confidence": float in [0,1], '
    '"justification": short string}. No prose.'
)


class LLMMatcher:
    """Adjudicates a borderline candidate pair via the tenant `LLMProvider`.

    Stateless w.r.t. tenant: `provider`/`config` are held for the call. Runs at
    temperature 0 for reproducibility. Malformed or unparseable output degrades
    to a zero-confidence `related_match` so the aligner drops it rather than
    crashing.
    """

    name = "llm"

    def __init__(self, provider: LLMProvider, config: ModelConfig) -> None:
        self._provider = provider
        self._config = config

    async def adjudicate(
        self, source: str, target: str, source_desc: str, target_desc: str
    ) -> LLMJudgement:
        user = (
            f"Source entity: {source}\nSource description: {source_desc or '(none)'}\n\n"
            f"Target entity: {target}\nTarget description: {target_desc or '(none)'}"
        )
        request = LLMRequest(
            model_config_id=self._config.id,
            messages=[
                Message(role="system", content=_LLM_SYSTEM),
                Message(role="user", content=user),
            ],
            params={"temperature": 0},
        )
        response = await self._provider.complete(self._config, request)
        return _parse_judgement(response.content)


def _parse_judgement(content: str) -> LLMJudgement:
    fallback = LLMJudgement(
        predicate=MappingRelation.RELATED_MATCH,
        confidence=0.0,
        justification="llm output could not be parsed",
    )
    raw = content.strip()
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback
    try:
        predicate = MappingRelation(str(data.get("predicate")))
    except ValueError:
        predicate = MappingRelation.RELATED_MATCH
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    justification = str(data.get("justification", "")) or "no justification provided"
    return LLMJudgement(predicate=predicate, confidence=confidence, justification=justification)
