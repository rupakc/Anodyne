"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, Select } from "@/components/ui/form";
import { ErrorAlert, InfoNote } from "@/components/ui/feedback";
import { EXPORT_FORMATS, type ApiClient, type ExportFormat, type ExportResult } from "@/lib/api";

/** Above this row count the gateway defaults an "auto" export to Parquet. */
const PARQUET_THRESHOLD = 500_000;

/**
 * Export one dataset version to a chosen format (or "auto"), surfacing the
 * presigned download URL the gateway returns. Mirrors the >500K-rows →
 * Parquet default the exporter applies server-side.
 */
export function ExportPanel({
  api,
  datasetId,
  versionId,
  rowCount,
}: {
  api: ApiClient;
  datasetId: string;
  versionId: string;
  rowCount: number;
}) {
  const [format, setFormat] = useState<ExportFormat | "">("");
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleExport() {
    setPending(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.exportVersion(datasetId, versionId, format || undefined);
      // Open the freshly-signed URL immediately so it can never go stale between
      // render and click. To download again the user re-runs the export (which
      // re-signs) — we never persist an href for a later click.
      window.open(res.url, "_blank", "noopener,noreferrer");
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed. Please try again.");
    } finally {
      setPending(false);
    }
  }

  const autoFormat = rowCount > PARQUET_THRESHOLD ? "parquet" : "csv";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Format" htmlFor={`export-format-${versionId}`} className="w-40">
          <Select
            id={`export-format-${versionId}`}
            value={format}
            onChange={(e) => setFormat(e.target.value as ExportFormat | "")}
          >
            <option value="">Auto ({autoFormat})</option>
            {EXPORT_FORMATS.map((f) => (
              <option key={f} value={f}>
                {f.toUpperCase()}
              </option>
            ))}
          </Select>
        </Field>
        <Button type="button" onClick={handleExport} disabled={pending}>
          {pending ? "Exporting…" : "Export"}
        </Button>
      </div>

      {format === "" ? (
        <InfoNote>
          Auto picks Parquet for versions over {PARQUET_THRESHOLD.toLocaleString()} rows, otherwise CSV.
          This version has {rowCount.toLocaleString()} rows → <strong>{autoFormat.toUpperCase()}</strong>.
        </InfoNote>
      ) : null}

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      {result ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm">
          <span>
            {result.artifact.format.toUpperCase()} export ready · download started ·{" "}
            {result.artifact.row_count.toLocaleString()} rows
          </span>
          <button
            type="button"
            onClick={handleExport}
            disabled={pending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-terracotta px-3 py-1.5 text-sm font-medium text-terracotta-foreground transition-colors hover:bg-terracotta/85 disabled:opacity-60"
          >
            <Download className="size-3.5" />
            Download again
          </button>
        </div>
      ) : null}
    </div>
  );
}
