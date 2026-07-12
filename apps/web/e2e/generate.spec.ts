import { test, expect } from "@playwright/test";

/**
 * @e2e happy-path: sign in → describe → review schema → generate → download.
 *
 * Drives the *full* local stack in a real browser (Keycloak, the
 * api-gateway, the Temporal generation-worker + Ray, and Ollama for the
 * schema-proposal/generation LLM calls) — see docs/dev-runbook.md for the
 * exact bring-up sequence this test assumes is already running:
 *
 *   make up  (Postgres/Redis/MinIO/Keycloak/Temporal/Ray/Ollama)
 *   make migrate && make seed
 *   register + pull an Ollama model for the demo tenant (docs/dev-runbook.md
 *     "Offline path" — the schema proposer 400s with "no model configured
 *     for this tenant" otherwise; see apps/api-gateway/src/api_gateway/deps.py)
 *   make dev  (api-gateway + generation-worker + `pnpm --dir apps/web dev`)
 *   pnpm --dir apps/web exec playwright install chromium   (one-time)
 *   pnpm --dir apps/web test:e2e
 *
 * Not run as part of `pnpm --dir apps/web test` (vitest only globs
 * `__tests__/**\/*.test.{ts,tsx}` — see vitest.config.mts) or the CI unit
 * lane; this is a separate, manually/CI-triggered `@e2e` lane.
 */
test.describe("generation happy path", () => {
  test(
    "sign in, create a dataset from a description, generate, and download the parquet",
    { tag: "@e2e" },
    async ({ page, context }) => {
      // Local LLM (Ollama) + Ray-backed generation can be slow, especially
      // on a cold model load — give this one test a generous ceiling well
      // above the config-level default.
      test.setTimeout(10 * 60 * 1000);

      // --- Sign in as the demo user (Keycloak's hosted login form) -------
      // /app/new requires a session (proxy.ts matcher: /app/:path*), so an
      // unauthenticated visit bounces through /login before landing on
      // Keycloak's own login page.
      await page.goto("/app/new");
      await page.getByRole("button", { name: "Sign in with Keycloak" }).click();

      await page.waitForURL(/\/realms\/anodyne\/.*\/auth/);
      await page.locator("#username").fill("demo@anodyne.dev");
      await page.locator("#password").fill("demo");
      await page.locator("#kc-login").click();

      // Auth.js's signIn() call always redirects to /app (see
      // app/login/page.tsx), regardless of the originally requested
      // /app/new — follow that link.
      await page.waitForURL(/\/app$/);
      await page.getByRole("link", { name: "Create a dataset" }).click();
      await page.waitForURL(/\/app\/new$/);

      // --- Step 1: describe the dataset -----------------------------------
      const datasetName = `E2E happy path ${Date.now()}`;
      await page.getByLabel("Dataset name").fill(datasetName);
      await page
        .getByLabel("Description")
        .fill(
          "Customer support tickets with an id, a customer email, a short " +
            "free-text subject, and a created timestamp.",
        );
      await page.getByLabel("Target row count").fill("25");
      await page.getByRole("button", { name: "Propose schema" }).click();

      // --- Step 2: review (accept) the proposed schema --------------------
      await expect(page.getByRole("heading", { name: "Review the proposed schema" })).toBeVisible({
        timeout: 60_000, // LLM-backed schema proposal round-trip
      });
      // Happy path: accept the proposal as-is, no field edits.
      await page.getByRole("button", { name: "Save & continue" }).click();

      // --- Step 3: confirm & generate --------------------------------------
      await expect(page.getByRole("heading", { name: "Confirm & generate" })).toBeVisible();
      await page.getByRole("button", { name: "Generate dataset" }).click();

      // --- Progress: wait for the job to reach "succeeded" -----------------
      await page.waitForURL(/\/app\/jobs\/[^/]+$/);
      await expect(page.getByText("Complete", { exact: true })).toBeVisible({
        timeout: 180_000, // Temporal/Ray generation run
      });
      await expect(page.getByText("Your dataset is ready.")).toBeVisible();

      const versionsLink = page.getByRole("link", { name: "View versions & download" });
      await versionsLink.click();
      await page.waitForURL(/\/app\/datasets\/[^/]+$/);

      // --- Download the generated Parquet artifact -------------------------
      await expect(page.getByRole("heading", { name: /^E2E happy path/ })).toBeVisible();
      const downloadButton = page.getByRole("button", { name: "Download" }).first();

      // Clicking "Download" calls `window.open(presignedUrl, "_blank", ...)`
      // (see DatasetVersions.handleDownload in dataset-versions.tsx). The
      // presigned MinIO URL carries no Content-Disposition header, so
      // Chromium treats the unrecognized/binary parquet payload as a
      // download on the new tab it opens, rather than trying to render it.
      const [popup] = await Promise.all([context.waitForEvent("page"), downloadButton.click()]);
      const download = await popup.waitForEvent("download");

      expect(download.suggestedFilename()).toMatch(/\.parquet$/);

      const downloadPath = await download.path();
      expect(downloadPath).not.toBeNull();
      const { statSync } = await import("node:fs");
      const stats = statSync(downloadPath!);
      expect(stats.size).toBeGreaterThan(0);
    },
  );
});
