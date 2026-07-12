import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    // Component tests (e.g. __tests__/wizard.test.tsx) opt into the DOM
    // environment per-file via a `// @vitest-environment jsdom` docblock,
    // so the default here stays "node" for the existing plain-logic tests.
    include: ["__tests__/**/*.test.{ts,tsx}"],
    server: {
      // `next-auth`'s main entry does `import ... from "next/server"`
      // (no extension). Next.js's package.json has no `exports` map, so
      // under Node's native ESM resolution (used when a dep is externalized
      // by Vitest's SSR pipeline) that extensionless specifier fails to
      // resolve. Forcing next-auth through Vite's own resolver (which does
      // resolve extensionless subpaths) sidesteps the issue.
      deps: {
        inline: ["next-auth"],
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
