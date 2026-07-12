# Plan — Sub-system G: Human-in-the-Loop & Annotation

See design: `docs/superpowers/specs/2026-07-12-hitl-annotation-g-design.md`.

1. `packages/anodyne-hitl/` scaffold (`pyproject.toml`, `src/anodyne_hitl/__init__.py`), register
   in root `pyproject.toml` workspace members + dev group + `apps/api-gateway/pyproject.toml`.
2. TDD `anodyne_hitl.models` — `ReviewKind`, `ReviewStatus`, `TargetType`/`Thumbs` literals,
   `Annotation`, `Feedback`, `ReviewTask`, `default_signal_name`.
3. TDD `anodyne_hitl.ports` — `AnnotationRepository`, `FeedbackRepository`, `ReviewRepository` ABCs.
4. `anodyne_storage/db.py` — add `annotations`, `feedback`, `review_tasks` tables +
   `_TENANT_TABLES` entries. Additive `DatasetRepository.get_version` port method +
   `SqlDatasetRepository.get_version` impl (TDD via `anodyne-storage`'s existing dataset_repo
   test file, new test function).
5. Migration `0008_hitl_annotations_feedback_review.py`, `down_revision="0007"`. Verify
   `alembic heads` → one head.
6. TDD `anodyne_hitl.registry` — `SqlAnnotationRepository`, `SqlFeedbackRepository`,
   `SqlReviewRepository` (structural mirror of `SqlEvaluationRepository`; no dedicated unit test,
   consistent with precedent — correctness covered by mypy/ruff + the route tests exercising them
   via fakes, and RLS via `test_rls.py`).
7. `apps/api-gateway/deps.py` — `get_annotation_repo`, `get_feedback_repo`, `get_review_repo`.
8. `anodyne_tenancy/authz.py` — `annotations:read/write`, `reviews:read/write` perms.
9. TDD `apps/api-gateway/src/api_gateway/hitl_routes.py` — all 7 endpoints + `apply_review_decision`
   helper; wire into `app.py` via one `include_router(build_hitl_router())` line.
10. `apps/api-gateway/tests/test_hitl_review_gate_integration.py` — real
    `GenerationWorkflow` parked at `wait_condition`, resumed via `apply_review_decision`
    (`@pytest.mark.integration`).
11. `packages/anodyne-storage/tests/test_rls.py` — extend for the 3 new tables
    (`@pytest.mark.integration`).
12. Self-review diff; run `ruff check --fix . && ruff format .`, `mypy .`,
    `pytest -m "not integration"` (report exact count), `alembic heads` from
    `packages/anodyne-storage`.
13. Commit frequently (conventional commits); final report per task instructions.
