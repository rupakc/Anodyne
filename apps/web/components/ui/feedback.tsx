import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Inline error banner (role="alert") — matches the pattern used throughout the app. */
export function ErrorAlert({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p
      role="alert"
      className={cn(
        "rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive",
        className,
      )}
    >
      {children}
    </p>
  );
}

/** Neutral informational note (e.g. the ">500K rows → Parquet" export hint). */
export function InfoNote({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p
      className={cn(
        "rounded-xl border border-amber/40 bg-amber/10 px-4 py-3 text-sm text-foreground",
        className,
      )}
    >
      {children}
    </p>
  );
}

/** Dashed empty-state box — matches the dataset/version empty states. */
export function EmptyState({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground",
        className,
      )}
    >
      {children}
    </div>
  );
}

/** Small animated loading line with an accessible label. */
export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <p className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
      <span
        aria-hidden
        className="size-3.5 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-terracotta"
      />
      {label}
    </p>
  );
}

/** Section eyebrow + heading pair used at the top of most views. */
export function SectionHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div>
        {eyebrow ? (
          <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            {eyebrow}
          </p>
        ) : null}
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance sm:text-3xl">
          {title}
        </h1>
        {description ? (
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground text-pretty">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-3">{actions}</div> : null}
    </div>
  );
}
