"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { createApiClient, type ApiClient, type JobStatus, type PerturbationJob } from "@/lib/api";
import { ErrorAlert, Loading } from "@/components/ui/feedback";

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  running: "Perturbing",
  awaiting_review: "Awaiting review",
  succeeded: "Complete",
  failed: "Failed",
};

const TERMINAL: readonly JobStatus[] = ["succeeded", "failed"];

export interface PerturbationViewProps {
  jobId: string;
  accessToken?: string;
  api?: ApiClient;
  pollIntervalMs?: number;
}

/**
 * Polls a perturbation job (`GET /perturbation-jobs/{id}`) to completion and,
 * on success, links to the derived version's dataset page.
 */
export function PerturbationView({
  jobId,
  accessToken,
  api: injectedApi,
  pollIntervalMs = 2000,
}: PerturbationViewProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [job, setJob] = useState<PerturbationJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | undefined;

    async function poll() {
      try {
        const next = await api.getPerturbationJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (TERMINAL.includes(next.status) && timer) {
          clearInterval(timer);
          timer = undefined;
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load the perturbation job.");
      }
    }

    void poll();
    timer = setInterval(poll, pollIntervalMs);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [api, jobId, pollIntervalMs]);

  if (error) return <ErrorAlert>{error}</ErrorAlert>;
  if (!job) return <Loading label="Loading perturbation…" />;

  const pct = Math.max(0, Math.min(100, Math.round(job.progress * 100)));
  const family = String((job.spec as { family?: string })?.family ?? "perturbation").replace("_", " ");

  return (
    <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            Perturbation {job.id.slice(0, 8)}
          </p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight capitalize text-balance">
            {family}
          </h1>
        </div>
        <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs font-medium tracking-wide uppercase">
          {STATUS_LABEL[job.status]}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        <div
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Perturbation progress"
          className="h-3 w-full overflow-hidden rounded-full bg-muted"
        >
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber to-terracotta transition-[width] duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex items-center justify-end text-xs text-muted-foreground">
          <span className="font-[family-name:var(--font-data)]">{pct}%</span>
        </div>
      </div>

      {job.message ? <p className="text-sm text-muted-foreground text-pretty">{job.message}</p> : null}

      {job.status === "failed" ? (
        <ErrorAlert>Perturbation failed. {job.message || "Please try again."}</ErrorAlert>
      ) : null}

      {job.status === "succeeded" ? (
        <div className="rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm">
          <p className="font-medium text-foreground">Derived version ready.</p>
          <p className="mt-1 text-muted-foreground">A new, perturbed version has been added to the dataset.</p>
          <Link
            href={`/app/datasets/${job.dataset_id}`}
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-terracotta px-4 py-2 text-sm font-medium text-terracotta-foreground transition-colors hover:bg-terracotta/85"
          >
            View versions
          </Link>
        </div>
      ) : null}
    </section>
  );
}
