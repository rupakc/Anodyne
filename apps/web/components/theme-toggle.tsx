"use client";

import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * Minimal light/dark toggle. Persists the choice in localStorage and falls
 * back to the OS preference; the flash-avoidance script in app/layout.tsx
 * sets the `dark` class on <html> before hydration.
 *
 * Deliberately stateless: which icon is visible is driven by CSS
 * (`dark:` variants keyed off the ancestor `.dark` class), not React state,
 * so there's no server/client mismatch and no effect needed to sync it.
 */
export function ThemeToggle() {
  function toggle() {
    const root = document.documentElement;
    const next = !root.classList.contains("dark");
    root.classList.toggle("dark", next);
    window.localStorage.setItem("anodyne-theme", next ? "dark" : "light");
  }

  return (
    <Button
      type="button"
      variant="outline"
      size="icon"
      aria-label="Toggle theme"
      onClick={toggle}
      className="border-border bg-card text-foreground hover:bg-secondary"
    >
      <Sun className="size-4 dark:hidden" />
      <Moon className="hidden size-4 dark:block" />
    </Button>
  );
}
