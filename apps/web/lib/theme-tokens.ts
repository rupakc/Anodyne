/**
 * Canonical list of the autumn-pastel brand tokens the design system exposes.
 * Kept as a single source of truth so the CSS (app/globals.css) and the
 * regression test (__tests__/theme.test.ts) can't silently drift apart.
 */
export const BRAND_TOKENS = [
  "amber",
  "amber-foreground",
  "terracotta",
  "terracotta-foreground",
  "dusty-rose",
  "dusty-rose-foreground",
  "sage",
  "sage-foreground",
  "cream",
  "cream-soft",
  "cream-foreground",
  "ink",
  "ink-muted",
] as const;

export const SEMANTIC_TOKENS = [
  "background",
  "foreground",
  "card",
  "card-foreground",
  "primary",
  "primary-foreground",
  "secondary",
  "secondary-foreground",
  "muted",
  "muted-foreground",
  "accent",
  "accent-foreground",
  "destructive",
  "destructive-foreground",
  "border",
  "input",
  "ring",
] as const;

export type BrandToken = (typeof BRAND_TOKENS)[number];
export type SemanticToken = (typeof SEMANTIC_TOKENS)[number];
