"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { ApiClient, ProviderConfig, ProviderKind } from "@/lib/api";
import { Field, Select } from "@/components/ui/form";

export interface ProviderSelectProps {
  api: ApiClient;
  kind: ProviderKind;
  label?: string;
  value: string;
  onChange: (id: string) => void;
  /** Include an explicit "auto / server default" empty option. */
  allowAuto?: boolean;
  id?: string;
}

/**
 * A modality-scoped provider picker backed by the tenant's registered
 * providers (`GET /{kind}`). Handles the "no provider configured" case with
 * a clear, actionable message linking to the provider registry — important
 * for image/audio/video, which need a GPU-backed provider on file.
 */
export function ProviderSelect({
  api,
  kind,
  label = "Provider",
  value,
  onChange,
  allowAuto,
  id = "provider-select",
}: ProviderSelectProps) {
  const [providers, setProviders] = useState<ProviderConfig[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.listProviders(kind).then(
      (list) => {
        if (cancelled) return;
        setProviders(list);
        setError(null);
      },
      (err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load providers.");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, kind]);

  if (error) {
    return (
      <p role="alert" className="text-sm text-destructive">
        {error}
      </p>
    );
  }

  if (providers === null) {
    return <p className="text-sm text-muted-foreground">Loading providers…</p>;
  }

  const enabled = providers.filter((p) => p.enabled);

  if (enabled.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-amber/50 bg-amber/10 px-4 py-3 text-sm">
        <p className="font-medium text-foreground">No provider configured for this modality yet.</p>
        <p className="mt-1 text-muted-foreground">
          Generation needs a registered provider (and, for image/audio/video, GPU capacity).{" "}
          <Link href="/app/providers" className="text-terracotta underline underline-offset-4">
            Register one
          </Link>{" "}
          to continue.
        </p>
      </div>
    );
  }

  return (
    <Field label={label} htmlFor={id}>
      <Select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {allowAuto ? <option value="">Auto (server default)</option> : null}
        {enabled.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} — {p.provider}/{p.model}
          </option>
        ))}
      </Select>
    </Field>
  );
}
