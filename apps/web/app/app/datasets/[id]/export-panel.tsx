"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, Select } from "@/components/ui/form";
import { ErrorAlert, InfoNote } from "@/components/ui/feedback";
import {
  EXPORT_FORMATS,
  GRAPH_EXPORT_FORMATS,
  type ApiClient,
  type ExportFormat,
  type GraphExportFormat,
  type Modality,
} from "@/lib/api";

/** Above this row count the gateway defaults an "auto" export to Parquet. */
const PARQUET_THRESHOLD = 500_000;

/**
 * Export one dataset version to a chosen format (or "auto"): the gateway
 * creates the export artifact, then its bytes are streamed straight through
 * an authenticated fetch and saved as a browser download (no presigned URL —
 * see `downloadToFile` in `lib/api.ts`). Mirrors the >500K-rows → Parquet
 * default the exporter applies server-side.
 */
export function ExportPanel({
  api,
  datasetId,
  versionId,
  rowCount,
  modality,
}: {
  api: ApiClient;
  datasetId: string;
  versionId: string;
  rowCount: number;
  /** When "graph", offer the graph interchange formats (Turtle, GraphML, Cypher, …). */
  modality?: Modality;
}) {
  const isGraph = modality === "graph";
  const [format, setFormat] = useState<ExportFormat | GraphExportFormat | "">(
    isGraph ? "node-link" : "",
  );
  const [pending, setPending] = useState(false);
  const [downloadedName, setDownloadedName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleExport() {
    setPending(true);
    setError(null);
    try {
      const filename = isGraph
        ? await api.exportGraphVersion(datasetId, versionId, (format || "node-link") as GraphExportFormat)
        : await api.exportVersion(datasetId, versionId, (format as ExportFormat) || undefined);
      setDownloadedName(filename);
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
        <Field label="Format" htmlFor={`export-format-${versionId}`} className="w-56">
          <Select
            id={`export-format-${versionId}`}
            value={format}
            onChange={(e) => setFormat(e.target.value as ExportFormat | GraphExportFormat | "")}
          >
            {isGraph ? (
              GRAPH_EXPORT_FORMATS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))
            ) : (
              <>
                <option value="">Auto ({autoFormat})</option>
                {EXPORT_FORMATS.map((f) => (
                  <option key={f} value={f}>
                    {f.toUpperCase()}
                  </option>
                ))}
              </>
            )}
          </Select>
        </Field>
        <Button type="button" onClick={handleExport} disabled={pending}>
          {pending ? "Exporting…" : "Export"}
        </Button>
      </div>

      {isGraph ? (
        <InfoNote>
          Graph exports project the property graph into RDF (Turtle/JSON-LD/OWL) or property-graph
          formats (GraphML, Cypher, Neo4j CSV). Large graphs stream chunked.
        </InfoNote>
      ) : format === "" ? (
        <InfoNote>
          Auto picks Parquet for versions over {PARQUET_THRESHOLD.toLocaleString()} rows, otherwise CSV.
          This version has {rowCount.toLocaleString()} rows → <strong>{autoFormat.toUpperCase()}</strong>.
        </InfoNote>
      ) : null}

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      {downloadedName ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm">
          <span>{downloadedName} · download started</span>
          <button
            type="button"
            onClick={handleExport}
            disabled={pending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-terracotta px-3 py-1.5 text-sm font-medium text-terracotta-foreground transition-colors hover:bg-terracotta/85 disabled:opacity-60"
          >
            <Download className="size-3.5" />
            {pending ? "Downloading…" : "Download"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
