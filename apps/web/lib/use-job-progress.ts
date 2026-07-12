"use client";

import { useEffect, useRef, useState } from "react";
import type { ApiClient, JobStatus } from "./api";

/**
 * Structural subset of the browser's `WebSocket` this hook actually uses —
 * lets tests inject a fake socket without extending the real DOM class (or
 * running under an environment that even has one).
 */
export interface WebSocketLike {
  onopen: (() => void) | null;
  onmessage: ((event: { data: string }) => void) | null;
  onerror: (() => void) | null;
  onclose: (() => void) | null;
  close(): void;
}

/** The JSON frame `anodyne_workflows.activities.set_status` publishes to `job:{id}`. */
interface JobProgressMessage {
  job_id: string;
  status: JobStatus;
  progress: number;
}

export interface JobProgressState {
  status: JobStatus;
  progress: number;
  message: string;
  /**
   * The job's owning dataset — only present on `GET /jobs/{id}` responses
   * (the initial snapshot and each poll), not on the WS frames themselves,
   * so it's populated on first fetch and simply carried forward after that.
   */
  datasetId: string | null;
  /** Where the current values came from — purely informational (e.g. a "live"/"reconnecting" badge). */
  connection: "connecting" | "live" | "polling";
}

export interface UseJobProgressOptions {
  jobId: string;
  /** Typed gateway client — used for the initial snapshot and the polling fallback. */
  api: ApiClient;
  /** Full `WS /jobs/{id}/stream` URL, e.g. from `jobStreamUrl()` in lib/api.ts. */
  wsUrl: string;
  /** Test seam: build the socket instead of `new WebSocket(url)`. */
  createSocket?: (url: string) => WebSocketLike;
  /** Poll cadence once the socket falls back. Defaults to 2s. */
  pollIntervalMs?: number;
}

const TERMINAL_STATUSES: readonly JobStatus[] = ["succeeded", "failed"];

function isTerminal(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}

/**
 * Live job progress: connects to `wsUrl` and applies each `{status,
 * progress}` frame as it arrives. If the socket errors, closes, or never
 * connects (e.g. the browser can't attach the `Authorization` header the
 * gateway's WS route requires — see docs/dev-runbook.md), it falls back to
 * polling `api.getJob(jobId)` on an interval, which does carry the header.
 * Polling stops once the job reaches a terminal status.
 */
export function useJobProgress({
  jobId,
  api,
  wsUrl,
  createSocket,
  pollIntervalMs = 2000,
}: UseJobProgressOptions): JobProgressState {
  const [state, setState] = useState<JobProgressState>({
    status: "pending",
    progress: 0,
    message: "Connecting…",
    datasetId: null,
    connection: "connecting",
  });

  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    let pollTimer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocketLike | undefined;

    function setConnection(connection: JobProgressState["connection"]) {
      if (!mountedRef.current) return;
      setState((prev) => ({ ...prev, connection }));
    }

    function applyJob(job: {
      status: JobStatus;
      progress: number;
      message?: string;
      dataset_id?: string;
    }) {
      if (!mountedRef.current) return;
      setState((prev) => ({
        status: job.status,
        progress: job.progress,
        message: job.message ?? prev.message,
        datasetId: job.dataset_id ?? prev.datasetId,
        connection: prev.connection,
      }));
    }

    function stopPolling() {
      if (pollTimer !== undefined) {
        clearInterval(pollTimer);
        pollTimer = undefined;
      }
    }

    function startPolling() {
      if (cancelled || pollTimer !== undefined) return;
      setConnection("polling");
      const poll = async () => {
        try {
          const job = await api.getJob(jobId);
          applyJob(job);
          if (isTerminal(job.status)) stopPolling();
        } catch {
          // Transient network hiccup: keep polling rather than crashing the view.
        }
      };
      void poll();
      pollTimer = setInterval(poll, pollIntervalMs);
    }

    // Seed an initial snapshot so a page refresh mid-job (or a job that's
    // already finished) shows real values before the socket has a chance to
    // say anything.
    void api
      .getJob(jobId)
      .then((job) => {
        if (!cancelled) applyJob(job);
      })
      .catch(() => {});

    try {
      const build = createSocket ?? ((url: string) => new WebSocket(url) as unknown as WebSocketLike);
      socket = build(wsUrl);
      socket.onopen = () => setConnection("live");
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as JobProgressMessage;
          applyJob(data);
          setConnection("live");
          if (isTerminal(data.status)) socket?.close();
        } catch {
          // Malformed frame: ignore rather than tear down the connection.
        }
      };
      socket.onerror = () => startPolling();
      socket.onclose = () => startPolling();
    } catch {
      startPolling();
    }

    return () => {
      cancelled = true;
      mountedRef.current = false;
      stopPolling();
      socket?.close();
    };
  }, [jobId, api, wsUrl, createSocket, pollIntervalMs]);

  return state;
}
