# Generation C6 — Template Catalog + Directives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or
> superpowers:executing-plans. Checkbox (`- [ ]`) tracking.

**Goal:** Starter template catalog (`GET /templates`, `POST /datasets/from-template`, web
affordance) + `GenerationDirective` schema + a `DirectiveGenerator` wrapper applied in the real
tabular generation path, all deterministic and unit-tested offline.

**Architecture:** New module `anodyne_dataset.directives` (schema); new module
`anodyne_generation.directives` (`DirectiveGenerator` wrapping `Generator`, applied post-hoc on the
`pyarrow.Table`); new package `anodyne-templates` (catalog + spec builder); one-line additive swap
in `anodyne_compute.ray_tasks`; additive gateway routes + `PATCH` field; additive web wizard toggle.

**Tech stack:** unchanged from C0 (Python 3.12, pydantic v2, pyarrow, numpy; Next.js/TS/vitest for web).

## Global constraints (mirrors C0)

- Register `anodyne-templates` in root `pyproject.toml` (dev group + `[tool.uv.sources]`); `uv sync`.
- `ruff` + `mypy --strict` clean; `uv run pytest -m "not integration and not e2e"` green after every task.
- Unique test basenames; no `tests/__init__.py`; importlib mode already set project-wide.
- Deterministic given a seed; commit per task; if web touched, keep its lint/typecheck/test/build green.

---

### Task 1: `anodyne_dataset.directives` — schema + parse/dump helpers

**Files:** create `packages/anodyne-dataset/src/anodyne_dataset/directives.py`; test
`packages/anodyne-dataset/tests/test_directives_models.py`.

- [ ] Write failing tests: `GenerationDirective` defaults (`rate=None`, `params={}`); construct one
  of each `DirectiveKind`; `parse_directives({})` → `[]`; `parse_directives({"directives": [...]})`
  round-trips through `dump_directives`; invalid `kind` string raises `pydantic.ValidationError`.
- [ ] Run → fail (`ModuleNotFoundError`).
- [ ] Implement `directives.py` per the design spec (`DirectiveKind` StrEnum; `GenerationDirective`
  BaseModel; `parse_directives`/`dump_directives` using `model_dump(mode="json")`/`model_validate`).
- [ ] Export `DirectiveKind`, `GenerationDirective`, `parse_directives`, `dump_directives` from
  `anodyne_dataset.directives` (no change needed to `anodyne_dataset.models`/`__init__.py` unless
  convenient — keep the existing `DatasetSpec.directives: dict[str, object]` field untouched).
- [ ] `uv run pytest packages/anodyne-dataset -q`, `ruff`, `mypy` clean.
- [ ] Commit: `feat(dataset): add GenerationDirective schema`.

---

### Task 2: `anodyne_generation.directives` — `DirectiveGenerator`

**Files:** create `packages/anodyne-generation/src/anodyne_generation/directives.py`; test
`packages/anodyne-generation/tests/test_directives_apply.py` and
`packages/anodyne-generation/tests/test_directives_generator.py`.

- [ ] Write failing tests (`test_directives_generator.py`):
  - No directives ⇒ `DirectiveGenerator(TabularSampler()).generate(spec, 0, 200, seed)` `.equals()`
    `TabularSampler().generate(spec, 0, 200, seed)` (byte-for-byte passthrough).
  - Same seed+directives ⇒ identical output (determinism).
  - Two disjoint shards (`start_row=0` vs `start_row=200`) with the same directive still differ
    (mirrors C0's `test_disjoint_ranges_differ`).
- [ ] Write failing tests (`test_directives_apply.py`):
  - `bias`: boolean field, no directive baseline proportion of `True` is ~50%; with
    `GenerationDirective(kind=BIAS, field="flag", value=True, rate=0.9)` over 500 rows, proportion
    of `True` ≥ 0.85 (allow small slack for the pre-existing baseline rate).
  - `edge_case` numeric: integer field `constraints={"min":0,"max":120}`;
    `GenerationDirective(kind=EDGE_CASE, field="age", value="max", rate=0.2)` over 300 rows ⇒
    ≥ 55 rows equal to 120 (0.2 × 300 = 60, allow for RNG-selection rounding).
  - `edge_case` null: nullable field; `value="null"` forces `None` in ≥ `rate` fraction of rows;
    non-nullable field + `value="null"` raises `DirectiveError`.
  - `use_case`: `GenerationDirective(kind=USE_CASE, name="rare_event", field="is_fraud",
    value=True)` (no explicit `rate`) resolves to the registry default (0.02) and the boolean
    column's `True` proportion is measurably above the ~50% baseline but low (bounded above by,
    say, 0.15) — confirms the default rate took effect, not a full bias.
  - Unknown `field` name (not in `spec.fields`) raises `DirectiveError` for `bias`/`edge_case`/
    `use_case`.
- [ ] Run → fail.
- [ ] Implement `directives.py`:
  - `DirectiveError(Exception)`.
  - `USE_CASE_DEFAULT_RATES = {"rare_event": 0.02, "balanced": 0.5, "high_risk_segment": 0.3}`.
  - `DirectiveGenerator(Generator)` — `__init__(self, inner: Generator)`; `generate(...)`:
    1. `table = self._inner.generate(spec, start_row, count, seed)`;
    2. `directives = parse_directives(spec.directives)`; if empty, return `table` unchanged;
    3. build `field_index = {f.name: f for f in spec.fields}` for validation;
    4. for `i, d` in `enumerate(directives)`: resolve `field = d.field` (validate against
       `field_index`, raise `DirectiveError` if missing), resolve `rate` (`d.rate` or, for
       `use_case`, `USE_CASE_DEFAULT_RATES[d.name]`), compute a boolean mask over `count` rows via
       `np.random.default_rng([seed, i, hash(field) & 0xFFFFFFFF, start_row]).random(count) <
       rate`, then overwrite the masked positions of that column with the resolved target value
       (symbolic `"min"`/`"max"` read from `field_index[field].constraints`; `"null"` requires
       `field_index[field].nullable` else `DirectiveError`; otherwise the literal `value`);
       rebuild the `pyarrow` column and replace it in the table (`table.set_column(...)`).
    5. return the mutated table.
- [ ] Run tests → pass; `ruff`/`mypy` clean.
- [ ] Commit: `feat(generation): add DirectiveGenerator applying bias/edge-case/use-case directives`.

---

### Task 3: Wire `DirectiveGenerator` into the real Ray path

**Files:** modify `packages/anodyne-compute/src/anodyne_compute/ray_tasks.py`; its existing test
file continues to pass unmodified (byte-identical for directive-free specs).

- [ ] Add a failing case to `packages/anodyne-compute/tests/test_ray_tasks.py` (or a new
  `test_ray_tasks_directives.py` if the existing file is integration-marked and this needs to stay
  unit-level): `generate_shard_bytes` on a spec **with** a `bias` directive produces a Parquet
  table whose column reflects the bias (parse via `pyarrow.parquet`); a directive-free spec's
  output is unchanged from before (regression guard).
- [ ] Implement: `from anodyne_generation.directives import DirectiveGenerator`; replace
  `TabularSampler().generate(...)` with `DirectiveGenerator(TabularSampler()).generate(...)` in
  `generate_shard_bytes`. Add `anodyne-generation`'s `directives` module has no new dependency (already
  a dependency of `anodyne-compute`).
- [ ] Run `anodyne-compute` tests (mark new ones appropriately if they need Ray/local mode) → pass.
- [ ] `ruff`/`mypy` clean.
- [ ] Commit: `feat(compute): apply generation directives in the tabular shard path`.

---

### Task 4: `anodyne-templates` — catalog + spec builder

**Files:** create `packages/anodyne-templates/pyproject.toml`,
`src/anodyne_templates/__init__.py`, `models.py`, `catalog.py`; test
`packages/anodyne-templates/tests/test_templates_catalog.py`; modify root `pyproject.toml`.

- [ ] Write failing tests: `list_templates()` returns ≥ 5 templates with unique `key`s covering
  `{"customers", "transactions", "support_tickets", "sensor_readings", "users_churn"}`; each has
  ≥ 1 field and a positive `default_target_rows`; `get_template("customers")` returns a match,
  `get_template("nope")` returns `None`; `build_dataset_spec(get_template("customers"),
  tenant_id=uuid4())` → `DatasetSpec` with `source == "template"`, non-empty `fields`,
  `status == "draft"`; passing `name=`/`target_rows=`/`directives=` overrides the template
  defaults; omitting them falls back to the template's.
- [ ] Run → fail.
- [ ] `pyproject.toml`: deps `["anodyne-dataset", "pydantic>=2.8"]` + workspace source.
- [ ] Implement `models.py` (`DatasetTemplate`) and `catalog.py` (`TEMPLATES` list with the 5
  templates below, `list_templates`, `get_template`, `build_dataset_spec`).
  - **customers**: `name`(name), `email`(email), `signup_date`(datetime), `plan`(categorical,
    choices=["free","pro","enterprise"]), `country`(address or categorical) — 1,000 rows default.
  - **transactions**: `transaction_id` not needed (id assigned by DB) — `amount`(float,
    min/max), `currency`(categorical), `timestamp`(datetime), `is_fraud`(boolean) — 5,000 rows.
  - **support_tickets**: `subject`(text), `priority`(categorical choices=["low","medium","high",
    "urgent"]), `status`(categorical choices=["open","pending","resolved","closed"]),
    `created_at`(datetime), `resolved`(boolean) — 2,000 rows.
  - **sensor_readings**: `sensor_id`(categorical or text), `temperature`(float),
    `humidity`(float), `reading_at`(datetime), `anomaly`(boolean) — 10,000 rows.
  - **users_churn**: `user_id` (implicit), `age`(integer), `tenure_months`(integer),
    `monthly_spend`(float), `plan`(categorical), `churned`(boolean, the label) — 3,000 rows;
    `default_directives` pre-populated with a `use_case` directive
    (`name="rare_event", field="churned", value=True`) so the "users+churn label" template
    demonstrates directive-driven class imbalance out of the box.
- [ ] Register in root `pyproject.toml` dev group + `[tool.uv.sources]`; `uv sync`.
- [ ] `uv run pytest packages/anodyne-templates -q`, `ruff`, `mypy` clean; full non-integration
  suite still green.
- [ ] Commit: `feat(templates): add starter dataset template catalog`.

---

### Task 5: Gateway — `GET /templates`, `POST /datasets/from-template`, `directives` on `PATCH`

**Files:** modify `apps/api-gateway/src/api_gateway/app.py`,
`apps/api-gateway/pyproject.toml` (+`anodyne-templates` dep/source); test
`apps/api-gateway/tests/test_templates_routes.py`.

- [ ] Write failing tests (reuse the `wired` fixture pattern from `test_dataset_routes.py`):
  - `GET /templates` (member/viewer both `datasets:read`) → 200, list includes `"customers"` key;
    no token → 401.
  - `POST /datasets/from-template` with `{"template_key": "customers"}` → 201, persisted spec has
    `source == "template"` and the template's fields; unknown key → 404; viewer → 403.
  - `POST /datasets/from-template` with `target_rows`/`directives` overrides → reflected in the
    persisted spec.
  - `PATCH /datasets/{id}` with `{"directives": {...}}` → 200, `repo.specs[id].directives` updated
    (existing `fields`/`name`/`target_rows` patch behavior unaffected — reuse
    `test_patch_updates_schema`'s dataset as a base case, extend or add a new test method).
- [ ] Run → fail.
- [ ] `pyproject.toml`: add `anodyne-templates` to deps + `[tool.uv.sources]`.
- [ ] Implement: `CreateFromTemplateRequest(BaseModel)` (`template_key: str`, `name: str | None`,
  `target_rows: int | None`, `directives: dict[str, object] | None`); `GET /templates` route
  (`datasets:read`) returning `[t.model_dump(mode="json") for t in list_templates()]`;
  `POST /datasets/from-template` (`datasets:write`) — `get_template`, 404 if `None`,
  `build_dataset_spec(...)`, `repo.create_spec`, return the spec; add `directives:
  dict[str, object] | None = None` to `UpdateDatasetRequest` and apply it in `update_dataset`
  alongside the existing optional fields.
- [ ] Run tests → pass; full non-integration suite green; `ruff`/`mypy` clean.
- [ ] Commit: `feat(gateway): add template catalog routes and directive patching`.

---

### Task 6: Root registration + full-suite sanity pass

- [ ] Confirm root `pyproject.toml` dev group + `[tool.uv.sources]` include `anodyne-templates`
  (done in Task 4, verified here alongside the gateway's new dependency on it).
- [ ] `uv sync && uv run pytest -q -m "not integration and not e2e"` — green, test count grown
  by Tasks 1–5's new tests.
- [ ] `uv run ruff check . && uv run mypy .` clean repo-wide.
- [ ] Commit if anything was missed: `chore: register anodyne-templates workspace-wide`.

---

### Task 7: Web — template step + wizard toggle (additive)

**Files:** modify `apps/web/lib/api.ts`; create `apps/web/app/app/new/template-step.tsx`; modify
`apps/web/app/app/new/wizard.tsx`; test `apps/web/__tests__/template-step.test.tsx`, extend
`apps/web/__tests__/wizard.test.tsx`.

- [ ] Invoke **frontend-design** skill guidance already established for the wizard (reuse existing
  autumn-pastel tokens/components — no new palette work).
- [ ] Write failing component test for `TemplateStep`: renders a list from a mocked
  `listTemplates`; selecting one calls `createFromTemplate({ template_key })` and the returned
  spec is surfaced via an `onCreated` callback.
- [ ] Write failing wizard test extension: a toggle control switches step 1 between "Describe" and
  "Start from a template"; selecting a template in the latter transitions straight to the
  `review` step showing the template's fields (mirrors the existing describe→review assertions);
  existing describe-path tests in `wizard.test.tsx` must still pass unmodified.
- [ ] Run → fail (missing exports/component).
- [ ] Implement: `lib/api.ts` — `DatasetTemplate` interface, `ApiClient.listTemplates()`,
  `ApiClient.createFromTemplate(input)`, wired into `createApiClient`. `template-step.tsx` —
  fetch-on-mount list, selectable cards/list styled like `ReviewStep`/`DescribeStep`. `wizard.tsx`
  — add `mode` state + toggle UI in the step-1 render branch; `handleTemplateSelected(spec)` sets
  `spec`/`fields`/`step("review")` exactly like `handleDescribe`'s success path.
- [ ] `pnpm --dir apps/web lint && pnpm --dir apps/web typecheck && pnpm --dir apps/web test &&
  pnpm --dir apps/web build` — all green.
- [ ] Commit: `feat(web): add start-from-template option to the create wizard`.

---

## Self-review

**Spec coverage:** directive schema (shared spine) → T1 ✓; directive application, wrapping not
rewriting `TabularSampler` → T2 ✓; real-path wiring → T3 ✓; template catalog package → T4 ✓;
gateway `GET /templates` (`datasets:read`) + `POST /datasets/from-template` (`datasets:write`) +
`directives` patch → T5 ✓; workspace registration → T6 ✓; web affordance → T7 ✓.

**Placeholders:** none — every task names exact files, functions, and test assertions; the 5
template field lists are enumerated in T4; the directive algorithm (seeded mask, symbolic
min/max/null) is fully specified in T2.

**Type/name consistency:** `GenerationDirective`/`DirectiveKind`/`parse_directives`/
`dump_directives` (T1) reused verbatim in T2/T5; `DirectiveGenerator` (T2) reused in T3;
`DatasetTemplate`/`list_templates`/`get_template`/`build_dataset_spec` (T4) reused in T5/T7;
route paths/RBAC permissions match the design spec's Gateway section exactly.

**Notes for execution:** run the full non-integration suite after every task (regression guard on
C0); T3's edit to `ray_tasks.py` is the only sibling-package-internal change and is one line plus
an import, consistent with "additive, minimal" for worker code.
