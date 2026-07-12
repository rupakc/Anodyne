"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { createApiClient, type ApiClient, type DatasetSpec, type DatasetVersion } from "@/lib/api";

export interface DatasetVersionsProps {
  datasetId: string;
  /** OIDC access token from `session.accessToken` (Task 11). */
  accessToken?: string;
  /** Dependency-injection point: pass a fake/mock `ApiClient` in tests. */
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error";

/**
 * Dataset detail: name/description plus every generated version, each with
 * a Download control that calls `GET .../versions/{id}/download` for a
 * presigned URL and opens it in a new tab.
 */
export function DatasetVersions({ datasetId, accessToken, api: injectedApi }: DatasetVersionsProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [dataset, setDataset] = useState<DatasetSpec | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getDataset(datasetId), api.listVersions(datasetId)])
      .then(([spec, list]) => {
        if (cancelled) return;
        setDataset(spec);
        setVersions(list);
        setState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load this dataset.");
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [api, datasetId]);

  async function handleDownload(versionId: string) {
    setDownloadingId(versionId);
    setDownloadError(null);
    try {
      const url = await api.downloadUrl(datasetId, versionId);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Failed to get a download link.");
    } finally {
      setDownloadingId(null);
    }
  }

  if (state === "loading") {
    return <p className="text-sm text-muted-foreground">Loading…</p>;
  }

  if (state === "error") {
    return (
      <p
        role="alert"
        className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
      >
        {error}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Dataset
        </p>
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          {dataset?.name}
        </h1>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">{dataset?.description}</p>
      </div>

      {downloadError ? (
        <p
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          {downloadError}
        </p>
      ) : null}

      <div>
        <h2 className="mb-3 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
          Versions
        </h2>
        {versions.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground">
            No versions yet — generation may still be running. Check back soon.
          </div>
        ) : (
          <ul className="flex flex-col gap-3" aria-label="Dataset versions">
            {versions.map((v) => (
              <li
                key={v.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 shadow-sm"
              >
                <div>
                  <p className="font-[family-name:var(--font-data)] text-sm font-medium">
                    {v.format.toUpperCase()} · {v.row_count.toLocaleString()} rows
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {new Date(v.created_at).toLocaleString()} · checksum {v.checksum.slice(0, 12)}…
                  </p>
                </div>
                <Button
                  type="button"
                  onClick={() => handleDownload(v.id)}
                  disabled={downloadingId === v.id}
                >
                  {downloadingId === v.id ? "Preparing…" : "Download"}
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
