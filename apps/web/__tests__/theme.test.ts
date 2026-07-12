import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BRAND_TOKENS, SEMANTIC_TOKENS } from "@/lib/theme-tokens";

const globalsCss = readFileSync(
  join(process.cwd(), "app", "globals.css"),
  "utf-8",
);

function rootBlock(selector: string): string {
  // Grabs the first `{ ... }` body following the given selector.
  const start = globalsCss.indexOf(`${selector} {`);
  expect(start, `expected to find a "${selector} {" block in globals.css`).toBeGreaterThan(-1);
  const end = globalsCss.indexOf("}", start);
  return globalsCss.slice(start, end);
}

describe("autumn-pastel theme tokens", () => {
  const lightRoot = rootBlock(":root");
  const darkRoot = rootBlock(".dark");

  it.each(BRAND_TOKENS)("defines brand token --%s in :root (light)", (token) => {
    expect(lightRoot).toMatch(new RegExp(`--${token}:\\s*`));
  });

  it.each(BRAND_TOKENS)("defines brand token --%s in .dark", (token) => {
    expect(darkRoot).toMatch(new RegExp(`--${token}:\\s*`));
  });

  it.each(SEMANTIC_TOKENS)("defines semantic role --%s in :root (light)", (token) => {
    expect(lightRoot).toMatch(new RegExp(`--${token}:\\s*`));
  });

  it.each(SEMANTIC_TOKENS)("defines semantic role --%s in .dark", (token) => {
    expect(darkRoot).toMatch(new RegExp(`--${token}:\\s*`));
  });

  it("exposes the brand palette as Tailwind color utilities via @theme inline", () => {
    for (const token of BRAND_TOKENS) {
      expect(globalsCss).toMatch(new RegExp(`--color-${token}:\\s*var\\(--${token}\\)`));
    }
  });

  it("light and dark use different values for every brand token (themes aren't just aliases)", () => {
    for (const token of BRAND_TOKENS) {
      const lightMatch = lightRoot.match(new RegExp(`--${token}:\\s*([^;]+);`));
      const darkMatch = darkRoot.match(new RegExp(`--${token}:\\s*([^;]+);`));
      expect(lightMatch).not.toBeNull();
      expect(darkMatch).not.toBeNull();
      expect(darkMatch?.[1]).not.toEqual(lightMatch?.[1]);
    }
  });
});
