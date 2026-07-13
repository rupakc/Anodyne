# Track GF — Ontology mapping / alignment

**Status: DONE.** Commit range `3082c0c..632cbb5` (4 commits) on `feat/graph-gf-mapping`.

## Files created
- `packages/anodyne-graph/src/anodyne_graph/mapping/models.py` — `MappingRelation`, `Mapping`, `MappingSet`.
- `.../mapping/ports.py` — `EntityMatcher` protocol, `EmbeddingFn` alias.
- `.../mapping/matchers.py` — `LexicalMatcher`, `EmbeddingMatcher`, `LLMMatcher`/`LLMJudgement`.
- `.../mapping/aligner.py` — `OntologyAligner`, `AlignmentThresholds`, HITL bridge.
- `.../mapping/sssom.py` — SSSOM (de)serializers + `PREDICATE_CURIE`.
- `.../mapping/__init__.py` — public API re-exports.
- 5 test files (`test_graph_mapping_{models,matchers,aligner,sssom,hitl}.py`).

## Files modified (outside anodyne-graph)
- `packages/anodyne-hitl/.../models.py:28` — added `ReviewKind.MAPPING_REVIEW`.
- `packages/anodyne-graph/pyproject.toml` — added `anodyne-hitl` dep.

## Public API (relied on by integrator/other tracks)
- `OntologyAligner(lexical, *, embedding=None, llm=None, thresholds=None, seed=0).align(source, target, *, source_id, target_id) -> MappingSet` (async).
- `AlignmentThresholds(auto_accept=0.85, review_floor=0.5, prefilter_floor=0.3, llm_top_k=25)`.
- `LexicalMatcher()`, `EmbeddingMatcher(embed_fn|None)`, `LLMMatcher(provider, config).adjudicate(...)`.
- `to_sssom_tsv/json(MappingSet)->bytes`, `from_sssom_tsv/json(bytes)->MappingSet`.
- `build_mapping_review_task(...)->ReviewTask|None`, `route_to_review(ms, repo, *, tenant_id, artifact_id, ...)` (async).

## HITL decision
Reused existing `review_tasks` schema; **no migration/table**. `kind` is an unconstrained `String` column (`db.py:289`), so adding the `MAPPING_REVIEW` enum value is additive-only. Chose **one ReviewTask per mapping-set** (lighter option) with `target_type="ontology_mapping_set"`, `target_id`=serialized SSSOM artifact id; flagged mappings live in the artifact.

## Deviations
- `EmbeddingMatcher` takes an injected `EmbeddingFn` (the `LLMProvider` port exposes no `embed`); degrades gracefully (`available=False`) when absent.
- Added `from_sssom_*` parsers (beyond spec's `to_*`) to satisfy the round-trip test.
- Prefilter operates on the *combined* lexical+embedding score so semantic-only matches survive.

## Final verification (worktree root)
- `pytest packages/anodyne-graph`: **122 passed, 1 warning** (28 new GF tests; warning is pre-existing rdflib deprecation in an untouched export test).
- `mypy .`: **Success: no issues found in 326 source files** (package-path invocation shows only pre-existing `anodyne_dataset` import-untyped noise; root is the real gate).
- `ruff check packages/anodyne-graph`: **All checks passed!** (`ruff format --check` clean).
