# Sub-system G — Human-in-the-Loop & Annotation

> Requirements 12 (human-in-the-loop), 13 (annotation + feedback). Spec → plan → TDD.
> Branch `feat/hitl-annotation-g`.

## 1. Goal

Two capabilities, generalized from patterns already in the repo:

1. **Annotation & feedback capture** (req 13): tenants label/tag/comment on dataset-version rows
   (`Annotation`) and rate dataset versions or evaluation runs (`Feedback`, incl. an expert
   override for judge dimensions).
2. **Review queue / HITL tasks** (req 12): a generic `ReviewTask` (pending/approved/rejected/
   changes_requested) that generalizes the existing `GenerationWorkflow.approve_schema` signal +
   `wait_condition` gate so any workflow needing sign-off can be paused/resumed the same way.

## 2. Where things live (hexagonal, matches repo precedent)

New package `packages/anodyne-hitl/` is a **bounded context**, modelled exactly on
`anodyne-evaluation` (own `models.py` + `ports.py` + `registry.py`; not folded into
`anodyne_dataset`, matching how evaluation — the other cross-cutting sub-system — got its own
package rather than growing `anodyne_dataset.ports` indefinitely). `anodyne-core` gets no new
imports.

- `anodyne_hitl/models.py` — `Annotation`, `Feedback`, `ReviewTask`, `ReviewKind`, `ReviewStatus`,
  `TargetType`, `Thumbs`, `default_signal_name(kind)`.
- `anodyne_hitl/ports.py` — `AnnotationRepository`, `FeedbackRepository`, `ReviewRepository` (ABCs).
- `anodyne_hitl/registry.py` — `SqlAnnotationRepository`, `SqlFeedbackRepository`,
  `SqlReviewRepository` (mirror `SqlEvaluationRepository`: import tables from
  `anodyne_storage.db`, `tenant_session` + explicit `tenant_id` filter).

Shared/touched: `anodyne_storage.db` (+3 tables + RLS entries), Alembic migration
`down_revision="0007"`, `anodyne_dataset.ports`/`dataset_repo.py` (additive `get_version`
lookup — see §4), `apps/api-gateway` (`hitl_routes.py` + `deps.py` + `authz.py` perms + `app.py`
one `include_router` line), root `pyproject.toml` + `apps/api-gateway/pyproject.toml`.

## 3. Domain models

```python
class ReviewKind(StrEnum):
    SCHEMA_APPROVAL = "schema_approval"
    DATASET_REVIEW = "dataset_review"
    EVALUATION_REVIEW = "evaluation_review"

class ReviewStatus(StrEnum):
    PENDING = "pending"; APPROVED = "approved"; REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"

class ReviewTask(BaseModel):
    id: UUID; tenant_id: UUID
    kind: ReviewKind
    target_type: str; target_id: UUID          # what is being reviewed
    workflow_id: str | None = None             # Temporal workflow to resume/abort
    signal_name: str | None = None             # defaults per kind if omitted
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_comment: str | None = None
    created_at: datetime; decided_at: datetime | None = None
```

`Annotation`: `id, tenant_id, dataset_id, version_id, row_index: int|None, record_id: str|None,
label: str|None, tags: list[str], comment: str|None, author: str, created_at`.

`Feedback`: `id, tenant_id, target_type: "dataset_version"|"evaluation_run", target_id: UUID,
rating: int|None (1-5), thumbs: "up"|"down"|None, comment: str|None,
expert_override: dict[str, object]|None, author: str, created_at`.

`default_signal_name({SCHEMA_APPROVAL: "approve_schema"}).get(kind)` — the one existing gate's
signal name, generalized as a lookup table rather than hardcoded per call site, so a new
workflow kind only needs one new dict entry.

## 4. Tables + migration

`annotations`, `feedback`, `review_tasks` — all `tenant_id` + RLS, added to
`anodyne_storage/db.py`'s shared `metadata` + `_TENANT_TABLES`, following the `0007` pattern
exactly (`metadata.create_all` for the new tables only, then enable+force RLS + policy).
Migration file `packages/anodyne-storage/src/anodyne_storage/migrations/versions/
0008_hitl_annotations_feedback_review.py`, `revision="0008"`, `down_revision="0007"` (current
head — confirmed via `alembic heads`).

Additive: `DatasetRepository.get_version(tenant_id, version_id) -> DatasetVersion | None` (new
abstract method + `SqlDatasetRepository` impl). Needed because `POST /feedback`'s target is a
bare `target_id` (no `dataset_id` in the URL to scope `list_versions`); every other consumer of
`DatasetRepository` in the repo is a duck-typed fake (not a real ABC subclass), so adding one
abstract method breaks nothing at runtime — `SqlDatasetRepository` is the only real subclass and
gets the implementation in the same change.

## 5. Generalizing the HITL gate — decision, and what's deliberately NOT changed

`GenerationWorkflow` (`anodyne_workflows/workflow.py`) already has the gate: `approve_schema`
signal + `wait_condition`. `apps/api-gateway/app.py`'s `start_generation` currently starts it
with `start_signal="approve_schema"` (already-approved — C0 does schema review *before* calling
generate; see the route's own comment: "the workflow's HITL gate itself stays intact for when
real pre-generate review lands"). `test_dataset_routes.py::test_generate_starts_workflow_and_
requires_write` asserts `call["start_signal"] == "approve_schema"` on every generate call across
~10 test functions in that file (and the audio/image/video dataset route test files), none of
which override a review-repo dependency.

**Decision:** do NOT touch `start_generation` or any existing dataset route. Flipping it to
create-a-pending-`ReviewTask`-and-start-without-`start_signal` is the natural follow-up once the
web UI's review screen (sub-system H) lands, but doing it now means either (a) adding a real DB
call unconditionally to a route none of ~10 existing tests mock the new dependency for — hangs/
fails offline — or (b) touching every one of those test files to add an override, which is
disproportionate blast radius for an additive sub-system and risks the hard "don't break
existing generation tests" constraint for no behavior change today (nothing currently produces a
pending schema-approval task before generate; the review already happened in the UI).

Instead, the mechanism is generalized and proven against the **real, unmodified**
`GenerationWorkflow`: `apply_review_decision(client, repo, review) -> ReviewTask` (in
`anodyne_hitl` is avoided to keep `temporalio` out of that package's import graph — kept as a
plain function in `api_gateway/hitl_routes.py`, using `Client` exactly like `perturbation_routes`/
`evaluation_routes` already inject it with no extra port wrapper) resolves `signal_name` (falling
back to `default_signal_name(kind)`), and:

- `approve` → `client.get_workflow_handle(workflow_id).signal(signal_name)`
- `reject` → `client.get_workflow_handle(workflow_id).cancel()` (generic abort — works for any
  workflow via Temporal's built-in cancellation, not just ones with a bespoke reject signal)
- `changes_requested` → no workflow action (status + comment persist; caller re-submits)

An `@pytest.mark.integration` test (`apps/api-gateway/tests/
test_hitl_review_gate_integration.py` — kept in `api-gateway`, not `anodyne-workflows`, since it
imports `api_gateway.hitl_routes.apply_review_decision`; a package must never depend on an app)
starts a real `GenerationWorkflow` *without*
`start_signal` (so it truly parks at `wait_condition`), creates a `ReviewTask(kind=
SCHEMA_APPROVAL, workflow_id=handle.id)`, calls `apply_review_decision(..., decision="approve")`,
and asserts the workflow completes — proving the generalized gate resumes the actual existing
workflow unmodified. Perturbation/evaluation workflows don't yet call `wait_condition` on
anything, so there's no gate to wire for them today; the mechanism (repo + decision endpoint +
per-kind signal default) is what "exposes it for when they do."

## 6. API routes (`api_gateway/hitl_routes.py`, `build_router()` factory like `evaluation_routes`)

Exactly the contract given (see task); tenant ownership: annotations resolve dataset+version via
`DatasetRepository.get_spec`/`get_version`; feedback resolves `dataset_version` via `get_version`
and `evaluation_run` via `EvaluationRepository.get_run`; reviews are tenant-filtered directly by
`ReviewRepository`. New perms in `authz.py`: `annotations:read/write`, `reviews:read/write` (read
in `_VIEWER`, write in `_MEMBER`, matching the existing shape).

## 7. Testing

Unit (offline, `--import-mode=importlib`, fake Temporal client + in-memory repos): model
validation, both Sql adapters need only mypy/ruff (their behavior is proven the same way
`SqlEvaluationRepository`'s is — no dedicated DB-backed unit test exists for it either; RLS is
proven in the single shared `test_rls.py` pattern — new tables added there under `@pytest.mark.
integration`), route tests per endpoint + permission + 404 ownership checks, and the review
decision → signal/cancel logic. Integration: the workflow-gate proof in §5, plus a `test_rls.py`
addition for the 3 new tables (`@pytest.mark.integration`, Docker).
