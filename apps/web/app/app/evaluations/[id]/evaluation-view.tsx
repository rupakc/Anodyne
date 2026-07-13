"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createApiClient,
  type ApiClient,
  type EvaluationReport,
  type EvaluationRun,
  type JobStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";
import { EvalReport } from "@/components/eval-report";

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  running: "Evaluating",
  awaiting_review: "Awaiting review",
  succeeded: "Complete",
  failed: "Failed",
};

export interface EvaluationViewProps {
  evaluationId: string;
  accessToken?: string;
  api?: ApiClient;
  /** Poll cadence for `GET /evaluations/{id}` while the run is in flight. */
  pollIntervalMs?: number;
}

/**
 * Live evaluation run: polls `GET /evaluations/{id}` until terminal, shows a
 * progress bar, then fetches and renders the LLM-as-a-Judge report on
 * success. (Evaluation has no WS stream — polling only, unlike generation.)
 */
export function EvaluationView({
  evaluationId,
  accessToken,
  api: injectedApi,
  pollIntervalMs = 2500,
}: EvaluationViewProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);

  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<"html" | "json" | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const reportFetched = useRef(false);
  const reportInFlight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    reportFetched.current = false;
    reportInFlight.current = false;

    // Fetch the report, retrying on the next poll if it fails: the report can
    // lag a succeeded run by a beat (the artifact is still being written), so a
    // single attempt would leave the user stuck on "Assembling…". We keep the
    // poll interval alive until the report actually lands, then stop cleanly.
    async function fetchReport() {
      if (reportFetched.current || reportInFlight.current) return;
      reportInFlight.current = true;
      try {
        const r = await api.getEvaluationReport(evaluationId);
        if (cancelled) return;
        reportFetched.current = true;
        setReport(r);
        setError(null);
        clearInterval(timer);
      } catch (err) {
        // Transient: leave `reportFetched` false so the next poll retries.
        if (!cancelled) {
          setError(
            err instanceof Error && err.message
              ? `${err.message} — retrying…`
              : "Couldn't load the report yet — retrying…",
          );
        }
      } finally {
        reportInFlight.current = false;
      }
    }

    async function poll() {
      try {
        const next = await api.getEvaluation(evaluationId);
        if (cancelled) return;
        setRun(next);
        if (next.status === "failed") {
          clearInterval(timer);
        } else if (next.status === "succeeded") {
          // Keep polling until the report is in hand; `fetchReport` clears the
          // interval on success. Terminal states other than success stop above.
          void fetchReport();
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load the evaluation.");
      }
    }

    const timer = setInterval(poll, pollIntervalMs);
    void poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [api, evaluationId, pollIntervalMs]);

  const handleDownload = useCallback(
    async (format: "html" | "json") => {
      setDownloading(format);
      setDownloadError(null);
      try {
        // Streams the report through the gateway and triggers a browser
        // download directly -- no presigned URL to open.
        await api.downloadReport(evaluationId, format);
      } catch (err) {
        setDownloadError(err instanceof Error ? err.message : "Failed to download the report.");
      } finally {
        setDownloading(null);
      }
    },
    [api, evaluationId],
  );

  if (error && !run) {
    return <ErrorAlert>{error}</ErrorAlert>;
  }

  if (!run) {
    return <Loading label="Loading evaluation…" />;
  }

  const pct = Math.max(0, Math.min(100, Math.round(run.progress * 100)));
  const isDone = run.status === "succeeded";
  const isFailed = run.status === "failed";

  return (
    <div className="flex flex-col gap-8">
      <SectionHeading
        eyebrow={`Evaluation ${evaluationId.slice(0, 8)}`}
        title="LLM-as-a-Judge evaluation"
        description="A panel of expert judges scores the dataset across fidelity, diversity, privacy, utility, bias, and qualitative dimensions."
        actions={
          isDone ? (
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => handleDownload("html")}
                disabled={downloading !== null}
              >
                {downloading === "html" ? "Downloading…" : "Download report (HTML)"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => handleDownload("json")}
                disabled={downloading !== null}
              >
                {downloading === "json" ? "Downloading…" : "Download report (JSON)"}
              </Button>
            </div>
          ) : undefined
        }
      />

      {!isDone ? (
        <section className="flex flex-col gap-3 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="rounded-full border border-amber/40 bg-amber/15 px-3 py-1 text-xs font-medium tracking-wide uppercase">
              {STATUS_LABEL[run.status]}
            </span>
            <span className="font-[family-name:var(--font-data)] text-xs text-muted-foreground">{pct}%</span>
          </div>
          <div
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Evaluation progress"
            className="h-3 w-full overflow-hidden rounded-full bg-muted"
          >
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber to-terracotta transition-[width] duration-500 ease-out"
              style={{ width: `${pct}%` }}
            />
          </div>
          {run.message ? <p className="text-sm text-muted-foreground text-pretty">{run.message}</p> : null}
        </section>
      ) : null}

      {isFailed ? (
        <ErrorAlert>Evaluation failed. {run.message || "Please try launching it again."}</ErrorAlert>
      ) : null}

      {downloadError ? <ErrorAlert>{downloadError}</ErrorAlert> : null}

      {isDone && !report && !error ? <Loading label="Assembling the report…" /> : null}
      {isDone && error ? <ErrorAlert>{error}</ErrorAlert> : null}
      {report ? <EvalReport report={report} accessToken={accessToken} /> : null}
    </div>
  );
}
