"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { ApiError, createApiClient, type ApiClient, type ReviewItem } from "@/lib/api";
import { EmptyState, ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";

export interface ReviewQueueProps {
  accessToken?: string;
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error" | "unavailable";

/** Small status pill; colored to hint approve/reject/pending without a shared dependency. */
function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "approved"
      ? "border-sage/50 bg-sage/20"
      : status === "rejected"
        ? "border-destructive/30 bg-destructive/10 text-destructive"
        : status === "changes_requested"
          ? "border-dusty-rose/40 bg-dusty-rose/15"
          : "border-amber/40 bg-amber/15";
  return (
    <span
      className={`rounded-full border px-2.5 py-1 font-[family-name:var(--font-data)] text-xs tracking-wide uppercase ${tone}`}
    >
      {status.replace(/_/g, " ")}
    </span>
  );
}

/**
 * Pending review queue (`GET /reviews?status=pending`). The HITL routes may
 * not be wired for a tenant yet, so a 404 degrades to a friendly empty state
 * rather than an error.
 */
export function ReviewQueue({ accessToken, api: injectedApi }: ReviewQueueProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.listReviews("pending").then(
      (list) => {
        if (cancelled) return;
        setItems(list);
        setState("ready");
      },
      (err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setState("unavailable");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load the review queue.");
        setState("error");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api]);

  return (
    <div className="flex flex-col gap-8">
      <SectionHeading
        eyebrow="Human review"
        title="Review queue"
        description="Approve, request changes on, or reject items awaiting a human decision — e.g. proposed schemas before generation runs."
      />

      {state === "loading" ? <Loading label="Loading the review queue…" /> : null}
      {state === "error" ? <ErrorAlert>{error}</ErrorAlert> : null}
      {state === "unavailable" ? (
        <EmptyState>Review workflows aren&apos;t enabled for this tenant yet.</EmptyState>
      ) : null}

      {state === "ready" && items.length === 0 ? (
        <EmptyState>Nothing pending — the queue is clear.</EmptyState>
      ) : null}

      {state === "ready" && items.length > 0 ? (
        <ul className="flex flex-col gap-3" aria-label="Pending reviews">
          {items.map((r) => (
            <li key={r.id}>
              <Link
                href={`/app/reviews/${r.id}`}
                className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 shadow-sm transition-colors hover:border-terracotta/40 hover:bg-muted/40"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="font-medium">
                      {r.title ?? r.kind ?? `Review ${r.id.slice(0, 8)}`}
                    </p>
                    <StatusBadge status={r.status} />
                  </div>
                  {r.summary ? (
                    <p className="mt-1 line-clamp-2 text-sm text-muted-foreground text-pretty">
                      {r.summary}
                    </p>
                  ) : null}
                  {r.target_type ? (
                    <p className="mt-1 font-[family-name:var(--font-data)] text-xs text-sage uppercase">
                      {r.target_type}
                      {r.target_id ? ` · ${r.target_id.slice(0, 8)}` : ""}
                    </p>
                  ) : null}
                </div>
                <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
              </Link>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
