# Generation Engine — tracked follow-ups

Non-blocking items deferred from C0–C6 (all reviewed and accepted as follow-ups). None regress
current behavior; capture here so they aren't lost.

## Functional
- **WebSocket progress auth.** `WS /jobs/{id}/stream` is header-auth only; browsers can't send
  `Authorization` on a WebSocket, so the UI falls back to HTTP polling (works). Add query-param /
  subprotocol token auth to the WS route + frontend, and keep the tenant-ownership check.
- **Default model selection.** `get_schema_proposer` (and per-modality provider factories) pick a
  tenant's *first* registered model/provider. Add an explicit "default model" flag per tenant.
- **Progress hook polish.** After a WS terminal message the hook fires one wasted poll and can leave
  a stuck "Reconnecting…" badge; guard `startPolling` when already terminal.
- **Web UI covers tabular-from-description + templates only.** No wizard yet for from-sample upload
  or text/image/audio/video (backend + API exist). Extend the UI per modality.
- **Live media runs need infra.** Image/audio/video self-hosted paths need a GPU node + a Ray actor
  holding real model weights; external paths need per-tenant provider API keys (registered via the
  `/image-providers` `/video-providers` `/audio-providers` routes). Only one external adapter per
  modality is implemented (OpenAI Images / ElevenLabs / Replicate); fal.ai/Runway are follow-ups
  behind the same port.
- **Tabular synth tuning.** CTGAN/TVAE default epochs (100) are untuned; deep/SDV fit-determinism
  relies on reseeding global RNGs (fine for CPU/small runs, not bulletproof under concurrent fits).
  Automatic synthesizer selection isn't implemented — chosen via `directives["synthesizer"]`.
- **Object-store per-tenant IAM.** Keys are correctly `{tenant}/…` after the C0 fix; if per-tenant
  S3 bucket/IAM prefix policies are later required, confirm the prefix scheme still fits.

## Cosmetic
- `packages/anodyne-workflows/src/anodyne_workflows/image_activities.py` docstring describes the
  abandoned inline-branch approach; rewrite for the modality-registry design.
- Structural asymmetry: image logic lives in `image_activities.py` while text/audio/video are inline
  in `handlers.py`; consider inlining image for consistency.
- `packages/anodyne-workflows/tests/test_video_activities.py` — filename leftover from the removed
  module; now tests the registry. Rename to `test_video_handler.py`.
- `apps/generation-worker` docstring says "five activities" (now more); the worker builds
  `FernetSecretStore` directly instead of via the `_secret_store()` helper with a friendly error.

## Test/CI
- RLS policies use `USING` without `WITH CHECK` (INSERT not RLS-constrained) — consistent across all
  tables incl. the original 0001/0002; add `WITH CHECK` if INSERT-time tenant enforcement is wanted.
- Third-party `StarletteDeprecationWarning` (httpx/testclient) appears in test output; filter it.
