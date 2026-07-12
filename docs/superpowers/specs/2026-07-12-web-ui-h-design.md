# Sub-system H — Web UI design note

Extends `apps/web` (Next.js 16, React 19, Tailwind v4, shadcn/base-ui, Auth.js/Keycloak)
so every backend capability on `main` is usable in the browser, in the autumn-pastel
design language. Branch: `feat/web-ui-h`.

## Routes / pages map

All authenticated routes live under `/app/*` (gated by `proxy.ts`) and are wrapped by a
new **app shell** (`app/app/layout.tsx` → `components/app-nav.tsx`) providing persistent
nav (Dashboard · Generate · Datasets · Reviews · Providers), theme toggle, and sign-out.

| Route | Page (server) | Client component | Purpose |
|---|---|---|---|
| `/app` | `app/page.tsx` | `dashboard.tsx` | Tenant home: stats, quick actions, recent datasets, pending reviews |
| `/app/new` | `new/page.tsx` | `generate-chooser.tsx` | Modality chooser → per-flow wizards |
| `/app/datasets` | `datasets/page.tsx` | `dataset-list.tsx` (existing) | Dataset browser |
| `/app/datasets/[id]` | `datasets/[id]/page.tsx` | `dataset-versions.tsx` | Versions + per-version actions + feedback |
| `/app/jobs/[id]` | existing | `job-progress-view.tsx` (existing) | Generation job progress (WS + poll) |
| `/app/perturbations/[id]` | `perturbations/[id]/page.tsx` | `perturbation-view.tsx` | Perturbation run progress (poll) |
| `/app/evaluations/[id]` | `evaluations/[id]/page.tsx` | `evaluation-view.tsx` + `components/eval-report.tsx` | Eval run progress + 360° report |
| `/app/providers` | `providers/page.tsx` | `providers-manager.tsx` | Per-modality provider registries |
| `/app/reviews` | `reviews/page.tsx` | `review-queue.tsx` | HITL pending queue |
| `/app/reviews/[id]` | `reviews/[id]/page.tsx` | `review-detail.tsx` | Approve / reject / request-changes |

### Generation flows (`generate-chooser.tsx`)
- **Tabular** → `wizard.tsx` (existing), extended with a 3rd source **From a sample**
  (`sample-step.tsx`): create (`source:"sample"`) → upload+profile → shared review → generate.
  Describe + Template sources unchanged.
- **Text** → `text-wizard.tsx`: task type (classification/QA/summarization/chat/generic)
  seeds the description; creates with `modality:"text"` → review → generate.
- **Image/Audio/Video** → `media-wizard.tsx`: provider-gated create + generate (no schema
  review). `components/provider-select.tsx` surfaces the "no provider/GPU configured" case.

## Component inventory (new)
- Shell: `app-nav.tsx`, `app/app/layout.tsx`, `dashboard.tsx`.
- Shared UI: `components/ui/form.tsx` (Field/TextInput/TextArea/Select),
  `components/ui/feedback.tsx` (ErrorAlert/InfoNote/EmptyState/Loading/SectionHeading),
  `components/provider-select.tsx`, `components/feedback-widget.tsx`.
- Generation: `new/{generate-chooser,sample-step,text-wizard,media-wizard}.tsx`.
- Dataset actions: `datasets/[id]/{export,perturb,evaluate,annotations}-panel.tsx`.
- Perturbation/Eval: `perturbation-view.tsx`, `evaluation-view.tsx`, `components/eval-report.tsx`.
- Providers/HITL: `providers-manager.tsx`, `review-queue.tsx`, `review-detail.tsx`.

## Backend endpoint mapping (all via `lib/api.ts` `ApiClient`)
- Providers: `GET/POST/DELETE /{models|image-providers|audio-providers|video-providers}` → `ModelConfig` (secret stripped).
- Generation: `POST /datasets` (source/modality), `POST /datasets/{id}/sample` (multipart), `POST /datasets/image`, `POST /datasets/audio`, `POST /datasets/{id}/generate` (`model_config_id`).
- Export: `POST /datasets/{id}/versions/{vid}/export?format=` → `{artifact,url}`; UI mirrors the >500K→Parquet auto default.
- Perturbation: `POST .../perturb` (family/intensity/target_fields/seed) → job with `result_version_id`; `GET /perturbation-jobs/{id}`, `GET /datasets/{id}/perturbation-jobs`.
- Evaluation: `POST .../evaluate` (optional reference version + fields) → run; `GET /evaluations/{id}`, `.../report`, `.../report/download`. Report renders expert scores (fidelity/diversity/privacy/utility/bias/qualitative), overall score, radar, rationale, recommendations.
- HITL (contract, not yet on `main`): `/reviews`, `/reviews/{id}`, `/reviews/{id}/decision`, `.../annotations`, `DELETE /annotations/{id}`, `POST /feedback`.

## Decisions
- **HITL routes are absent on `main`** (confirmed): all review/annotation/feedback calls
  are best-effort — a 404/501 degrades to a friendly "not enabled" empty state, never an error.
- **Video has no dedicated create route**; `media-wizard` uses the generic `POST /datasets`
  with `modality:"video"` and surfaces backend errors gracefully (labelled a preview).
- **Perturbation/Evaluation have no WS stream** → poll `GET` endpoints (`pollIntervalMs`
  injectable for tests). Generation keeps its existing WS-with-poll-fallback hook.
- **Test contract stability**: `__tests__/mock-api.ts` `baseMockApi()` is the single source
  for `ApiClient` stubs so extending the interface never breaks unrelated tests.
- Sample-step file input uses `aria-required` (not `required`) so the JS gate — not jsdom's
  file-input constraint validation — governs submit.

## Verification
- Unit: 106 vitest tests pass (was 81; +25). Build (`next build`), `eslint`, `tsc --noEmit` all clean.
- e2e: `e2e/generate.spec.ts` updated for the chooser/dashboard nav and extended with a CSV
  export step (full-stack `@e2e` lane; unchanged run model per `playwright.config.ts`).
