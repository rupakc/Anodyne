"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { createApiClient, type ApiClient, type DatasetSpec } from "@/lib/api";

export interface DatasetListProps {
  /** OIDC access token from `session.accessToken` (Task 11). */
  accessToken?: string;
  /** Dependency-injection point: pass a fake/mock `ApiClient` in tests. */
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error";

/** `GET /datasets` rendered as a browsable, linkable list. */
export function DatasetList({ accessToken, api: injectedApi }: DatasetListProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [datasets, setDatasets] = useState<DatasetSpec[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listDatasets()
      .then((list) => {
        if (cancelled) return;
        setDatasets(list);
        setState("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load datasets.");
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [api]);

  if (state === "loading") {
    return <p className="text-sm text-muted-foreground">Loading datasets…</p>;
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

  if (datasets.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card p-8 text-center text-sm text-muted-foreground">
        No datasets yet.{" "}
        <Link href="/app/new" className="text-terracotta underline underline-offset-4">
          Create one
        </Link>{" "}
        to get started.
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-3" aria-label="Datasets">
      {datasets.map((d) => (
        <li key={d.id}>
          <Link
            href={`/app/datasets/${d.id}`}
            className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-5 shadow-sm transition-colors hover:border-terracotta/40 hover:bg-muted/40 sm:flex-row sm:items-center sm:justify-between"
          >
            <div>
              <p className="font-medium text-foreground">{d.name}</p>
              <p className="mt-1 line-clamp-1 text-sm text-muted-foreground text-pretty">{d.description}</p>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="rounded-full border border-border bg-background px-2.5 py-1 font-[family-name:var(--font-data)] tracking-wide uppercase">
                {d.status}
              </span>
              <span className="font-[family-name:var(--font-data)]">
                {d.target_rows.toLocaleString()} rows
              </span>
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
