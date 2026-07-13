"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { LogOut, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/app", label: "Dashboard", exact: true },
  { href: "/app/new", label: "Generate", exact: false },
  { href: "/app/datasets", label: "Datasets", exact: false },
  { href: "/app/reviews", label: "Reviews", exact: false },
  { href: "/app/providers", label: "Providers", exact: false },
] as const;

function isActive(pathname: string, href: string, exact: boolean): boolean {
  return exact ? pathname === href : pathname === href || pathname.startsWith(`${href}/`);
}

/**
 * Persistent top navigation for the authenticated app surface. Highlights
 * the active section (client-side via `usePathname`), exposes the theme
 * toggle, and posts sign-out through the server action handed down from the
 * server-component layout.
 */
export function AppNav({
  userLabel,
  signOutAction,
}: {
  userLabel: string;
  signOutAction: () => void | Promise<void>;
}) {
  const pathname = usePathname() ?? "/app";
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <div className="flex items-center gap-6">
          <Link
            href="/app"
            className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight"
          >
            anodyne
          </Link>
          <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
            {LINKS.map((l) => {
              const active = isActive(pathname, l.href, l.exact);
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
                    active
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  {l.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden max-w-[14rem] truncate text-xs text-muted-foreground sm:inline">
            {userLabel}
          </span>
          <ThemeToggle />
          <form action={signOutAction} className="hidden md:block">
            <Button type="submit" variant="outline" size="sm" aria-label="Sign out">
              <LogOut className="size-3.5" data-icon="inline-start" />
              Sign out
            </Button>
          </form>
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="md:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-4" /> : <Menu className="size-4" />}
          </Button>
        </div>
      </div>

      {open ? (
        <nav
          className="border-t border-border bg-background px-6 py-3 md:hidden"
          aria-label="Primary mobile"
        >
          <ul className="flex flex-col gap-1">
            {LINKS.map((l) => {
              const active = isActive(pathname, l.href, l.exact);
              return (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    onClick={() => setOpen(false)}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "block rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                      active ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-muted",
                    )}
                  >
                    {l.label}
                  </Link>
                </li>
              );
            })}
            <li className="mt-1">
              <form action={signOutAction}>
                <Button type="submit" variant="outline" size="sm" className="w-full">
                  <LogOut className="size-3.5" data-icon="inline-start" />
                  Sign out
                </Button>
              </form>
            </li>
          </ul>
        </nav>
      ) : null}
    </header>
  );
}
