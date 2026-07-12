"use client";

import { useMemo } from "react";
import Link from "next/link";
import { createApiClient, jobStreamUrl, type ApiClient, type JobStatus } from "@/lib/api";
import { useJobProgress, type WebSocketLike } from "@/lib/use-job-progress";

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  running: "Generating",
  awaiting_review: "Awaiting review",
  succeeded: "Complete",
  failed: "Failed",
};

const STATUS_STYLE: Record<JobStatus, string> = {
  pending: "border-border bg-muted text-muted-foreground",
  running: "border-amber/40 bg-amber/15 text-ink",
  awaiting_review: "border-dusty-rose/40 bg-dusty-rose/15 text-ink",
  succeeded: "border-sage/50 bg-sage/20 text-ink",
  failed: "border-destructive/30 bg-destructive/10 text-destructive",
};

const CONNECTION_LABEL = {
  connecting: "Connecting…",
  live: "Live",
  polling: "Reconnecting… (polling)",
} as const;

export interface JobProgressViewProps {
  jobId: string;
  /** OIDC access token from `session.accessToken` (Task 11). */
  accessToken?: string;
  /** Dependency-injection seams for tests — bypass real network/socket. */
  api?: ApiClient;
  wsUrl?: string;
  createSocket?: (url: string) => WebSocketLike;
}

/**
 * Live progress for one generation job. Connects to `WS /jobs/{id}/stream`
 * for real-time `{status, progress}` frames and falls back to polling
 * `GET /jobs/{id}` if the socket never opens or drops (see
 * `lib/use-job-progress.ts` for why: browsers can't attach the gateway's
 * required `Authorization` header to a WebSocket handshake).
 */
export function JobProgressView({
  jobId,
  accessToken,
  api: injectedApi,
  wsUrl: injectedWsUrl,
  createSocket,
}: JobProgressViewProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const wsUrl = useMemo(() => injectedWsUrl ?? jobStreamUrl(jobId), [injectedWsUrl, jobId]);

  const { status, progress, message, datasetId, connection } = useJobProgress({
    jobId,
    api,
    wsUrl,
    createSocket,
  });

  const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));

  return (
    <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            Job {jobId.slice(0, 8)}
          </p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
            Generation progress
          </h1>
        </div>
        <span
          className={
            "rounded-full border px-3 py-1 text-xs font-medium tracking-wide uppercase " +
            STATUS_STYLE[status]
          }
        >
          {STATUS_LABEL[status]}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        <div
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Generation progress"
          className="h-3 w-full overflow-hidden rounded-full bg-muted"
        >
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber to-terracotta transition-[width] duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span data-testid="connection-status">{CONNECTION_LABEL[connection]}</span>
          <span className="font-[family-name:var(--font-data)]">{pct}%</span>
        </div>
      </div>

      <p className="text-sm text-muted-foreground text-pretty">{message}</p>

      {status === "failed" ? (
        <p
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          Generation failed. {message || "Please try again from the dataset's page."}
        </p>
      ) : null}

      {status === "succeeded" ? (
        <div className="rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm">
          <p className="font-medium text-foreground">Your dataset is ready.</p>
          <p className="mt-1 text-muted-foreground">
            The generated rows have been assembled into a downloadable version.
          </p>
          {datasetId ? (
            <Link
              href={`/app/datasets/${datasetId}`}
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-terracotta px-4 py-2 text-sm font-medium text-terracotta-foreground transition-colors hover:bg-terracotta/85"
            >
              View versions &amp; download
            </Link>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
