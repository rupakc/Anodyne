# Anodyne — Generation C6 (Starter Template Catalog + Directives) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system C, spec C6 (final slice)
- **Depends on:** [Generation Engine architecture](./2026-07-12-generation-engine-architecture-design.md) · Generation C0 (on `main`)

## Goal

Two additions on top of the C0 tabular vertical slice, both pure-logic and fully unit-testable
offline:

1. A **starter template catalog** — ready-made `DatasetSpec` blueprints for common use-cases a
   user can pick and customize instead of writing a description from scratch.
2. **`GenerationDirective`** handling (requirement 4) — declarative steering on a `DatasetSpec`
   that biases toward subpopulations, targets a named use-case, or forces edge-case/rare rows,
   applied deterministically at generation time.

## Decisions

| Decision | Choice |
|---|---|
| Directive schema location | `anodyne_dataset.directives` (the shared spine — architecture doc requirement 4 says directives must be honorable by other modalities later; keeping the schema in `anodyne-dataset` next to `DatasetSpec` lets C2–C5 import it without depending on `anodyne-generation`) |
| `DatasetSpec.directives` wire format | Unchanged `dict[str, object]` (no breaking change to C0's model/API/storage/web types) — it holds `{"directives": [<GenerationDirective.model_dump()>, ...]}`, parsed/dumped via `parse_directives`/`dump_directives` helpers |
| Directive kinds | `bias` (skew a field toward a value at a target rate), `edge_case` (force a field to an extreme/rare value at a target rate), `use_case` (named preset resolving to a `bias`/`edge_case` with a sensible default rate, e.g. `rare_event` → 2%) |
| Application point | New `DirectiveGenerator` in `anodyne-generation` **wraps** a `Generator` (default: `TabularSampler`) post-processing its output table — `TabularSampler` itself is untouched, honoring "wrap or compose, don't rewrite" |
| Wiring into the real path | `anodyne_compute.ray_tasks.generate_shard_bytes` swaps its bare `TabularSampler()` call for `DirectiveGenerator(TabularSampler())` — a one-line, additive change |
| Determinism | Directive row-selection uses `np.random.default_rng([seed, directive_index, field_hash, start_row])` — independent per directive/shard, reproducible given the same seed+shard |
| New package | `anodyne-templates` — models + a static catalog + a `DatasetSpec` builder; depends only on `anodyne-dataset` |
| Templates catalog | 5 starter templates: customers, transactions, support tickets, sensor readings, users+churn label |
| Gateway surface | `GET /templates` (`datasets:read`) lists the catalog; `POST /datasets/from-template` (`datasets:write`) builds+persists a `DatasetSpec` from a template (bypasses the LLM proposer entirely — fields come from the template); `PATCH /datasets/{id}` gains an optional `directives` field so any dataset (from either source) can have directives attached/edited |
| Web UI | The create wizard's first step gains a two-tab toggle ("Describe" / "Start from a template"); selecting a template calls `POST /datasets/from-template` and joins the existing review → confirm pipeline unchanged |

## Components

### 1. `anodyne_dataset.directives` (new module in the existing `anodyne-dataset` package)

```python
class DirectiveKind(StrEnum):
    BIAS = "bias"
    EDGE_CASE = "edge_case"
    USE_CASE = "use_case"

class GenerationDirective(BaseModel):
    kind: DirectiveKind
    field: str | None = None        # target field name (required for bias/edge_case)
    value: object | None = None     # target value ("min"/"max"/"null" symbolic for edge_case)
    rate: float | None = None       # fraction of rows affected, 0..1 (None -> kind default)
    name: str | None = None         # use_case preset name (required for kind=use_case)
    params: dict[str, object] = Field(default_factory=dict)

def parse_directives(raw: dict[str, object]) -> list[GenerationDirective]
def dump_directives(directives: list[GenerationDirective]) -> dict[str, object]
```

Pure Pydantic + two pure functions — no generation logic, no new dependency for `anodyne-dataset`.

### 2. `anodyne_generation.directives` (new module in the existing `anodyne-generation` package)

- `USE_CASE_DEFAULT_RATES: dict[str, float]` — small built-in registry (`rare_event`: 0.02,
  `balanced`: 0.5, `high_risk_segment`: 0.3) resolving a `use_case` directive's effective rate
  when `rate` is omitted; `use_case` directives otherwise apply exactly like `bias`.
- `DirectiveError(Exception)` — raised for a directive referencing an unknown field or a
  malformed edge/bias value; the gateway maps it the same way `SchemaProposalError` is mapped
  (400, client-fixable).
- `DirectiveGenerator(Generator)` — wraps an inner `Generator`:
  1. calls `inner.generate(spec, start_row, count, seed)` for the baseline table (untouched
     `TabularSampler` behavior when `spec.directives` is empty — verified equal-by-value);
  2. `parse_directives(spec.directives)`;
  3. for each directive, in order, forces/skews `rate` fraction of the shard's rows for
     `field` (or the `use_case` preset's resolved field) toward `value`/target, using an
     independent seeded RNG stream per `(seed, directive_index, field, start_row)` to pick which
     row indices are affected — deterministic, reproducible;
  4. returns the mutated `pyarrow.Table`.
- Numeric fields: `value` may be a literal, or the symbols `"min"`/`"max"` resolved from the
  field's own `constraints`. Categorical/boolean fields: `value` is a literal from `choices`
  (bias) or any forced value (edge_case, e.g. forcing a rare category not in normal `choices`).
  Nullable fields: `value == "null"` forces `None` (raises `DirectiveError` if the field isn't
  `nullable`).

### 3. `anodyne-templates` (new package)

- `models.py`: `DatasetTemplate` — `key`, `name`, `description`, `category`, `modality`,
  `fields: list[FieldSpec]`, `default_target_rows`, `default_directives: dict[str, object]`.
- `catalog.py`: `TEMPLATES: list[DatasetTemplate]` (customers, transactions, support_tickets,
  sensor_readings, users_churn); `list_templates() -> list[DatasetTemplate]`;
  `get_template(key: str) -> DatasetTemplate | None`; `build_dataset_spec(template, *, tenant_id,
  name=None, target_rows=None, directives=None) -> DatasetSpec` (`source="template"`).

### 4. Gateway (`apps/api-gateway`, additive)

- `GET /templates` (`datasets:read`) → `[t.model_dump(mode="json") for t in list_templates()]`.
- `POST /datasets/from-template` (`datasets:write`) — body `{template_key, name?, target_rows?,
  directives?}`; 404 on unknown `template_key`; persists via the existing `DatasetRepository`.
- `UpdateDatasetRequest` gains `directives: dict[str, object] | None`; `PATCH /datasets/{id}`
  applies it like the existing `name`/`target_rows`/`fields` patches.
- `anodyne_generation.directives.DirectiveError` mapped to 400 wherever a directive is
  validated eagerly is out of scope for C6 (directives are validated lazily at generation time,
  same as C0's constraint handling) — no new eager-validation route added.

### 5. Ray path (`anodyne-compute`, one-line additive change)

`generate_shard_bytes` builds `DirectiveGenerator(TabularSampler())` instead of a bare
`TabularSampler()`. Behavior is identical when `spec.directives` is empty (existing C0 tests for
`generate_shard_bytes`/`remote_generate_shard` keep passing unchanged).

### 6. Web UI (`apps/web`, additive)

- `lib/api.ts`: `DatasetTemplate` type, `ApiClient.listTemplates()`, `ApiClient.createFromTemplate()`.
- `app/app/new/template-step.tsx`: fetches the catalog, renders a selectable list
  (name/description/category), on select calls `createFromTemplate` and joins the wizard at the
  `review` step exactly like `handleDescribe` does today.
- `wizard.tsx`: a `mode: "describe" | "template"` toggle (default `"describe"`) renders either
  `DescribeStep` or `TemplateStep` for step 1; steps 2–3 (`review`, `confirm`) are unchanged.

## Testing strategy

- **Unit (`anodyne-dataset`):** `test_directives_models.py` — `GenerationDirective` defaults/
  validation; `parse_directives`/`dump_directives` round-trip; unknown-kind rejection.
- **Unit (`anodyne-generation`):** `test_directives_apply.py` — a `bias` directive measurably
  shifts a categorical/boolean column's proportion toward the target above a threshold vs. the
  undirected baseline; an `edge_case` directive produces the exact targeted value in ≥ `rate`
  fraction of rows (numeric `min`/`max`, categorical forced value, `null` on a nullable field);
  a `use_case` directive resolves its default rate and applies like `bias`; unknown field / null
  on non-nullable field raise `DirectiveError`. `test_directives_generator.py` — `DirectiveGenerator`
  is a byte-for-byte passthrough of the inner generator with no directives; deterministic given
  the same seed; two disjoint shards remain disjoint (mirrors C0's sampler determinism tests).
- **Unit (`anodyne-templates`):** `test_templates_catalog.py` — ≥ 5 unique template keys covering
  the required use-cases; `get_template` miss → `None`; `build_dataset_spec` produces a valid
  `DatasetSpec` with `source == "template"`, merges/override defaults correctly.
- **Gateway:** `test_templates_routes.py` — `GET /templates` returns the catalog and is
  `datasets:read`-gated; `POST /datasets/from-template` creates+persists a spec, 404s on an
  unknown key, is `datasets:write`-gated; `PATCH /datasets/{id}` with `directives` persists them.
- **Ray path:** extend the existing `anodyne-compute` integration test (or add a unit-level
  check) confirming `generate_shard_bytes` output is unchanged for a directive-free spec and
  reflects directive effects for one with directives.
- **Web:** component test for `TemplateStep` (mocked `listTemplates`/`createFromTemplate`) and an
  extension of `wizard.test.tsx` covering the toggle → template selection → review path; existing
  describe-path tests must keep passing unchanged.

## Non-goals (C6)

Text/image/audio/video honoring directives (reserved for C2–C5 to consume the same
`GenerationDirective` schema); a directive-authoring UI beyond attaching `directives` via the
existing `PATCH`; SDV-specific bias hooks (C1 concern); server-side eager directive validation at
create/patch time (validated lazily at generation, consistent with C0's field-constraint handling).
