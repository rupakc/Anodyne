"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight, Database, GitBranch, Sparkles, ClipboardCheck } from "lucide-react";
import { createApiClient, type ApiClient, type DatasetSpec, type ReviewItem } from "@/lib/api";
import { ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";

export interface DashboardProps {
  accessToken?: string;
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error";

const QUICK_ACTIONS = [
  { href: "/app/new", label: "Generate a dataset", icon: Sparkles },
  { href: "/app/datasets", label: "Browse datasets", icon: Database },
  { href: "/app/reviews", label: "Review queue", icon: ClipboardCheck },
  { href: "/app/providers", label: "Manage providers", icon: GitBranch },
] as const;

/**
 * Tenant home. Pulls the dataset roster (`GET /datasets`) and the pending
 * review queue (`GET /reviews?status=pending`, tolerated as empty if the
 * HITL routes aren't wired yet), and surfaces quick actions into every
 * major flow.
 */
export function Dashboard({ accessToken, api: injectedApi }: DashboardProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [datasets, setDatasets] = useState<DatasetSpec[]>([]);
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const list = await api.listDatasets();
        if (cancelled) return;
        setDatasets(list);
        setState("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load your workspace.");
        setState("error");
        return;
      }
      // Reviews are best-effort: the HITL routes may not exist yet.
      try {
        const pending = await api.listReviews("pending");
        if (!cancelled) setReviews(pending);
      } catch {
        if (!cancelled) setReviews([]);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [api]);

  const readyCount = datasets.filter((d) => d.status === "ready").length;

  return (
    <main className="mx-auto flex min-h-full w-full max-w-6xl flex-1 flex-col gap-10 px-6 py-10">
      <SectionHeading
        eyebrow="Workspace"
        title="Your synthetic-data workspace"
        description="Generate across every modality, perturb and evaluate versions, and clear the review queue — all from here."
        actions={
          <Button render={<Link href="/app/new" />} nativeButton={false} size="lg">
            <Sparkles className="size-4" data-icon="inline-start" />
            New dataset
          </Button>
        }
      />

      {state === "error" ? <ErrorAlert>{error}</ErrorAlert> : null}

      <section aria-label="Summary" className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Datasets" value={state === "loading" ? "—" : datasets.length} />
        <Stat label="Ready" value={state === "loading" ? "—" : readyCount} />
        <Stat label="Pending reviews" value={state === "loading" ? "—" : reviews.length} />
        <Stat label="Modalities" value={5} hint="tabular · text · image · audio · video" />
      </section>

      <section aria-label="Quick actions">
        <h2 className="mb-3 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
          Jump in
        </h2>
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {QUICK_ACTIONS.map((a) => (
            <li key={a.href}>
              <Link
                href={a.href}
                className="group flex h-full items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 shadow-sm transition-colors hover:border-terracotta/40 hover:bg-muted/40"
              >
                <span className="flex items-center gap-3 text-sm font-medium">
                  <a.icon className="size-5 text-terracotta" />
                  {a.label}
                </span>
                <ArrowRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section aria-label="Recent datasets">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Recent datasets
          </h2>
          <Link href="/app/datasets" className="text-sm text-terracotta underline-offset-4 hover:underline">
            View all
          </Link>
        </div>
        {state === "loading" ? (
          <Loading label="Loading your workspace…" />
        ) : datasets.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground">
            No datasets yet.{" "}
            <Link href="/app/new" className="text-terracotta underline underline-offset-4">
              Generate your first
            </Link>
            .
          </div>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2">
            {datasets.slice(0, 6).map((d) => (
              <li key={d.id}>
                <Link
                  href={`/app/datasets/${d.id}`}
                  className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-5 shadow-sm transition-colors hover:border-terracotta/40 hover:bg-muted/40"
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-medium">{d.name}</p>
                    <span className="rounded-full border border-border bg-background px-2 py-0.5 font-[family-name:var(--font-data)] text-[0.65rem] tracking-wide text-muted-foreground uppercase">
                      {d.modality}
                    </span>
                  </div>
                  <p className="line-clamp-2 text-sm text-muted-foreground text-pretty">{d.description}</p>
                  <p className="font-[family-name:var(--font-data)] text-xs text-sage uppercase">
                    {d.status} · {d.target_rows.toLocaleString()} rows
                  </p>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      {reviews.length > 0 ? (
        <section aria-label="Pending reviews">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
              Awaiting your review
            </h2>
            <Link href="/app/reviews" className="text-sm text-terracotta underline-offset-4 hover:underline">
              Open queue
            </Link>
          </div>
          <ul className="flex flex-col gap-2">
            {reviews.slice(0, 4).map((r) => (
              <li key={r.id}>
                <Link
                  href={`/app/reviews/${r.id}`}
                  className="flex items-center justify-between gap-3 rounded-xl border border-dusty-rose/40 bg-dusty-rose/10 px-4 py-3 text-sm transition-colors hover:bg-dusty-rose/20"
                >
                  <span className="font-medium">{r.title ?? r.kind ?? `Review ${r.id.slice(0, 8)}`}</span>
                  <ArrowRight className="size-4 text-muted-foreground" />
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <p className="font-[family-name:var(--font-display)] text-3xl font-semibold tracking-tight">{value}</p>
      <p className="mt-1 text-xs tracking-wide text-muted-foreground uppercase">{label}</p>
      {hint ? <p className="mt-1 text-[0.7rem] text-muted-foreground">{hint}</p> : null}
    </div>
  );
}
