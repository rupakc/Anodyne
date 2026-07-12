import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for the `@e2e` happy-path suite (Task 14).
 *
 * This suite drives the *full* local stack in a real browser — Keycloak
 * login, the create-from-description wizard, generation, and a Parquet
 * download — so it is a separate, manually/CI-triggered lane, not part of
 * the `pnpm --dir apps/web test` (vitest) unit lane:
 *
 *   - vitest only picks up `__tests__/**\/*.test.{ts,tsx}` (see
 *     vitest.config.mts's `test.include`), so anything under `e2e/` is
 *     invisible to it regardless of this file.
 *   - This config deliberately does NOT define a `webServer` block, so
 *     `playwright test` never tries to boot Next.js (or anything else)
 *     itself. The developer/CI job is responsible for having the whole
 *     stack already running before invoking it — see docs/dev-runbook.md
 *     for the exact sequence (`make up` → migrate/seed → register an Ollama
 *     model for the demo tenant → `make dev` → `playwright install chromium`
 *     → `pnpm --dir apps/web test:e2e`).
 */
export default defineConfig({
  testDir: "./e2e",
  // Keep the whole suite tagged so `playwright test --grep @e2e` (or CI's
  // default invocation) only ever runs these happy-path specs.
  grep: /@e2e/,
  timeout: 120_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
