"""`OntologyAligner`: orchestrate the hybrid matcher stack into a `MappingSet`.

Pipeline per entity kind (node types + edge types):

1. **prefilter** — score every source/target candidate pair; keep those whose
   combined score clears `prefilter_floor` (cheap: embeddings are cached per
   label). Combined = lexical, or the mean of lexical + embedding when an
   embedding matcher is available (so a purely semantic match is not discarded
   by a low lexical score).
2. **rerank** — pick each source entity's best target by combined score, with a
   stable tie-break on `(subject_id, object_id)`.
3. **LLM adjudication** — the borderline best-candidates (combined in
   `[review_floor, auto_accept)`) are sent, top-k first, to the `LLMMatcher`,
   which returns the predicate + confidence + justification.
4. **route** — accept `>= auto_accept`; flag `needs_review` in
   `[review_floor, auto_accept)`; drop below `review_floor`.

Everything is deterministic: sorted iteration, stable tie-breaks; the `seed` is
recorded in metadata for reproducibility (there is no stochastic step). LLM and
embedding access is only through the injected matchers.

Also hosts the HITL bridge (`build_mapping_review_task` / `route_to_review`):
low-confidence mappings become a single `anodyne_hitl.ReviewTask` referencing
the serialized mapping-set artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from anodyne_hitl.models import ReviewKind, ReviewTask
from anodyne_hitl.ports import ReviewRepository
from pydantic import BaseModel

from anodyne_graph.mapping.matchers import EmbeddingMatcher, LexicalMatcher, LLMMatcher
from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet
from anodyne_graph.models import EdgeType, GraphOntology, NodeType

# `target_type` recorded on the HITL ReviewTask for an alignment review; the
# `target_id` is the persisted mapping-set artifact's id.
MAPPING_SET_TARGET_TYPE = "ontology_mapping_set"


class AlignmentThresholds(BaseModel):
    """Confidence bands + prefilter/LLM budget for an alignment run."""

    auto_accept: float = 0.85
    review_floor: float = 0.5
    prefilter_floor: float = 0.3
    llm_top_k: int = 25


@dataclass
class _Candidate:
    subject_id: str
    object_id: str
    combined: float
    matcher: str
    subject_desc: str
    object_desc: str


def _node_desc(nt: NodeType) -> str:
    props = ", ".join(sorted(p.name for p in nt.properties))
    return f"node type with properties: {props or '(none)'}"


def _edge_desc(et: EdgeType) -> str:
    props = ", ".join(sorted(p.name for p in et.properties))
    return f"relation from {et.source_type} to {et.target_type}; properties: {props or '(none)'}"


def _predicate_for_score(score: float) -> MappingRelation:
    if score >= 0.95:
        return MappingRelation.EXACT_MATCH
    if score >= 0.8:
        return MappingRelation.CLOSE_MATCH
    return MappingRelation.RELATED_MATCH


class OntologyAligner:
    """Aligns a source `GraphOntology` to a target one, emitting a `MappingSet`."""

    def __init__(
        self,
        lexical: LexicalMatcher,
        *,
        embedding: EmbeddingMatcher | None = None,
        llm: LLMMatcher | None = None,
        thresholds: AlignmentThresholds | None = None,
        seed: int = 0,
    ) -> None:
        self._lexical = lexical
        self._embedding = embedding if (embedding and embedding.available) else None
        self._llm = llm
        self._thresholds = thresholds or AlignmentThresholds()
        self._seed = seed

    async def align(
        self,
        source: GraphOntology,
        target: GraphOntology,
        *,
        source_id: str,
        target_id: str,
    ) -> MappingSet:
        candidates: list[_Candidate] = []
        candidates += self._best_candidates(
            [(nt.name, _node_desc(nt)) for nt in source.node_types],
            [(nt.name, _node_desc(nt)) for nt in target.node_types],
        )
        candidates += self._best_candidates(
            [(et.name, _edge_desc(et)) for et in source.edge_types],
            [(et.name, _edge_desc(et)) for et in target.edge_types],
        )

        t = self._thresholds
        # LLM adjudication on the top-k borderline best-candidates.
        adjudicated: dict[tuple[str, str], tuple[MappingRelation, float, str]] = {}
        if self._llm is not None:
            borderline = sorted(
                (c for c in candidates if t.review_floor <= c.combined < t.auto_accept),
                key=lambda c: (-c.combined, c.subject_id, c.object_id),
            )[: t.llm_top_k]
            for c in borderline:
                j = await self._llm.adjudicate(
                    c.subject_id, c.object_id, c.subject_desc, c.object_desc
                )
                adjudicated[(c.subject_id, c.object_id)] = (
                    j.predicate,
                    j.confidence,
                    j.justification,
                )

        mappings: list[Mapping] = []
        for c in candidates:
            key = (c.subject_id, c.object_id)
            if key in adjudicated:
                predicate, confidence, justification = adjudicated[key]
                matcher = "llm"
            else:
                confidence = c.combined
                predicate = _predicate_for_score(confidence)
                justification = f"{c.matcher} similarity {confidence:.3f}"
                matcher = c.matcher
            if confidence < t.review_floor:
                continue  # dropped
            mappings.append(
                Mapping(
                    subject_id=c.subject_id,
                    predicate=predicate,
                    object_id=c.object_id,
                    confidence=confidence,
                    justification=justification,
                    matcher=matcher,
                    subject_label=c.subject_id,
                    object_label=c.object_id,
                    needs_review=confidence < t.auto_accept,
                )
            )

        mappings.sort(key=lambda m: (m.subject_id, m.object_id))
        return MappingSet(
            source_ontology_id=source_id,
            target_ontology_id=target_id,
            mappings=mappings,
            metadata={
                "seed": self._seed,
                "auto_accept": t.auto_accept,
                "review_floor": t.review_floor,
                "prefilter_floor": t.prefilter_floor,
                "embedding": self._embedding is not None,
                "llm": self._llm is not None,
                "needs_review_count": sum(1 for m in mappings if m.needs_review),
            },
        )

    def _best_candidates(
        self, sources: list[tuple[str, str]], targets: list[tuple[str, str]]
    ) -> list[_Candidate]:
        """Best target per source entity, above the prefilter floor."""
        out: list[_Candidate] = []
        matcher_name = "lexical+embedding" if self._embedding is not None else "lexical"
        for s_name, s_desc in sorted(sources):
            scored: list[tuple[float, str, str]] = []
            for t_name, t_desc in sorted(targets):
                lex = self._lexical.score(s_name, t_name)
                if self._embedding is not None:
                    combined = 0.5 * lex + 0.5 * self._embedding.score(s_name, t_name)
                else:
                    combined = lex
                if combined >= self._thresholds.prefilter_floor:
                    scored.append((combined, t_name, t_desc))
            if not scored:
                continue
            scored.sort(key=lambda x: (-x[0], x[1]))
            combined, t_name, t_desc = scored[0]
            out.append(
                _Candidate(
                    subject_id=s_name,
                    object_id=t_name,
                    combined=combined,
                    matcher=matcher_name,
                    subject_desc=s_desc,
                    object_desc=t_desc,
                )
            )
        return out


def build_mapping_review_task(
    mapping_set: MappingSet,
    *,
    tenant_id: UUID,
    artifact_id: UUID,
    workflow_id: str | None = None,
    signal_name: str | None = None,
    task_id: UUID | None = None,
) -> ReviewTask | None:
    """Build one HITL `ReviewTask` for a mapping-set's flagged mappings.

    Returns `None` when nothing is flagged `needs_review`. We create a *single*
    task per mapping-set (the lighter option in the plan) that references the
    serialized mapping-set artifact (`target_id=artifact_id`); the specific
    flagged mappings live in that artifact rather than fanning out one task per
    mapping. `workflow_id`/`signal_name` wire an approve/reject decision back to
    a paused workflow when applicable.
    """
    flagged = [m for m in mapping_set.mappings if m.needs_review]
    if not flagged:
        return None
    return ReviewTask(
        id=task_id or uuid4(),
        tenant_id=tenant_id,
        kind=ReviewKind.MAPPING_REVIEW,
        target_type=MAPPING_SET_TARGET_TYPE,
        target_id=artifact_id,
        workflow_id=workflow_id,
        signal_name=signal_name,
    )


async def route_to_review(
    mapping_set: MappingSet,
    repo: ReviewRepository,
    *,
    tenant_id: UUID,
    artifact_id: UUID,
    workflow_id: str | None = None,
    signal_name: str | None = None,
    task_id: UUID | None = None,
) -> ReviewTask | None:
    """Persist a `ReviewTask` for the flagged mappings via `repo`, if any.

    Returns the created task (already persisted) or `None` when nothing needs
    review. The `MappingSet` itself is expected to have been serialized to the
    object store as `artifact_id` (e.g. via `sssom.to_sssom_json`) by the caller.
    """
    task = build_mapping_review_task(
        mapping_set,
        tenant_id=tenant_id,
        artifact_id=artifact_id,
        workflow_id=workflow_id,
        signal_name=signal_name,
        task_id=task_id,
    )
    if task is None:
        return None
    await repo.create(task)
    return task
