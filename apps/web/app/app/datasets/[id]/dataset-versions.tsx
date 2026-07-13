"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Download, FileDown, GitBranch, Gauge, Tags, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";
import { FeedbackWidget } from "@/components/feedback-widget";
import {
  createApiClient,
  type ApiClient,
  type DatasetSpec,
  type DatasetVersion,
  type GraphArtifact,
  type GraphOntology,
  type PerturbationJob,
} from "@/lib/api";
import { OntologyView } from "@/components/ontology-view";
import { GraphExplorer } from "@/components/graph-explorer";
import { ExportPanel } from "./export-panel";
import { PerturbPanel } from "./perturb-panel";
import { EvaluatePanel } from "./evaluate-panel";
import { AnnotationsPanel } from "./annotations-panel";

export interface DatasetVersionsProps {
  datasetId: string;
  /** OIDC access token from `session.accessToken` (Task 11). */
  accessToken?: string;
  /** Dependency-injection point: pass a fake/mock `ApiClient` in tests. */
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error";
type PanelKey = "export" | "perturb" | "evaluate" | "annotate" | "visualize";

/**
 * Above this many graph elements (nodes + edges — the version `row_count`) we
 * refuse to fetch and parse the full node-link JSON in the browser: a large
 * artifact would OOM the tab before it ever reached the explorer's 120-node
 * sampler. Below it, the artifact is small enough to fetch and render.
 */
const GRAPH_VISUALIZE_MAX_ELEMENTS = 5_000;

/** Reads the proposed ontology from a graph dataset's directives, tolerating shape drift. */
function ontologyOf(spec: DatasetSpec | null): GraphOntology | null {
  const raw = (spec?.directives as { ontology?: GraphOntology } | undefined)?.ontology;
  return raw && Array.isArray(raw.node_types) ? raw : null;
}

/**
 * Dataset detail: name/description, a feedback control, every generated
 * version with its full action surface (download, export, perturb, evaluate,
 * annotate), and any perturbation runs launched from those versions.
 */
export function DatasetVersions({ datasetId, accessToken, api: injectedApi }: DatasetVersionsProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [dataset, setDataset] = useState<DatasetSpec | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [perturbations, setPerturbations] = useState<PerturbationJob[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [openPanel, setOpenPanel] = useState<{ versionId: string; panel: PanelKey } | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [spec, list] = await Promise.all([
          api.getDataset(datasetId),
          api.listVersions(datasetId),
        ]);
        if (cancelled) return;
        setDataset(spec);
        setVersions(list);
        setState("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load this dataset.");
        setState("error");
        return;
      }
      // Perturbation history is best-effort (route may be absent).
      try {
        const jobs = (await api.listPerturbationJobs(datasetId)) ?? [];
        if (!cancelled) setPerturbations(jobs);
      } catch {
        if (!cancelled) setPerturbations([]);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [api, datasetId]);

  async function handleDownload(versionId: string) {
    setDownloadingId(versionId);
    setDownloadError(null);
    try {
      // Streams the artifact through the gateway and triggers a browser
      // download directly -- no presigned URL to open.
      await api.downloadVersionFile(datasetId, versionId);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Failed to download this version.");
    } finally {
      setDownloadingId(null);
    }
  }

  function togglePanel(versionId: string, panel: PanelKey) {
    setOpenPanel((prev) =>
      prev && prev.versionId === versionId && prev.panel === panel ? null : { versionId, panel },
    );
  }

  if (state === "loading") return <Loading label="Loading dataset…" />;
  if (state === "error") return <ErrorAlert>{error}</ErrorAlert>;

  const fieldNames = dataset?.fields.map((f) => f.name) ?? [];
  const isGraph = dataset?.modality === "graph";
  const ontology = isGraph ? ontologyOf(dataset) : null;

  return (
    <div className="flex flex-col gap-8">
      <SectionHeading eyebrow="Dataset" title={dataset?.name ?? ""} description={dataset?.description} />

      <FeedbackWidget targetType="dataset" targetId={datasetId} accessToken={accessToken} api={injectedApi} />

      {downloadError ? <ErrorAlert>{downloadError}</ErrorAlert> : null}

      {isGraph && ontology ? (
        <div>
          <h2 className="mb-3 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Ontology
          </h2>
          <OntologyView ontology={ontology} />
        </div>
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
            {versions.map((v) => {
              const open = openPanel?.versionId === v.id ? openPanel.panel : null;
              return (
                <li key={v.id} className="rounded-2xl border border-border bg-card p-5 shadow-sm">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="flex items-center gap-2 font-[family-name:var(--font-data)] text-sm font-medium">
                        {v.format.toUpperCase()} · {v.row_count.toLocaleString()} rows
                        {v.parent_version_id ? (
                          <span className="inline-flex items-center gap-1 rounded-full border border-dusty-rose/40 bg-dusty-rose/15 px-2 py-0.5 text-[0.65rem] tracking-wide uppercase">
                            <GitBranch className="size-3" /> derived
                          </span>
                        ) : null}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {new Date(v.created_at).toLocaleString()} · checksum {v.checksum.slice(0, 12)}…
                      </p>
                    </div>
                    <Button type="button" onClick={() => handleDownload(v.id)} disabled={downloadingId === v.id}>
                      <Download className="size-3.5" data-icon="inline-start" />
                      {downloadingId === v.id ? "Downloading…" : "Download"}
                    </Button>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2 border-t border-border pt-3">
                    {isGraph ? (
                      <ActionButton active={open === "visualize"} onClick={() => togglePanel(v.id, "visualize")} icon={Share2}>
                        Visualize
                      </ActionButton>
                    ) : null}
                    <ActionButton active={open === "export"} onClick={() => togglePanel(v.id, "export")} icon={FileDown}>
                      Export
                    </ActionButton>
                    <ActionButton active={open === "perturb"} onClick={() => togglePanel(v.id, "perturb")} icon={GitBranch}>
                      Perturb
                    </ActionButton>
                    <ActionButton active={open === "evaluate"} onClick={() => togglePanel(v.id, "evaluate")} icon={Gauge}>
                      Evaluate
                    </ActionButton>
                    <ActionButton active={open === "annotate"} onClick={() => togglePanel(v.id, "annotate")} icon={Tags}>
                      Annotate
                    </ActionButton>
                  </div>

                  {open ? (
                    <div className="mt-4 rounded-xl border border-border bg-muted/30 p-4">
                      {open === "visualize" ? (
                        <VisualizePanel
                          api={api}
                          datasetId={datasetId}
                          versionId={v.id}
                          rowCount={v.row_count}
                        />
                      ) : null}
                      {open === "export" ? (
                        <ExportPanel
                          api={api}
                          datasetId={datasetId}
                          versionId={v.id}
                          rowCount={v.row_count}
                          modality={dataset?.modality}
                        />
                      ) : null}
                      {open === "perturb" ? (
                        <PerturbPanel api={api} datasetId={datasetId} versionId={v.id} fieldNames={fieldNames} />
                      ) : null}
                      {open === "evaluate" ? (
                        <EvaluatePanel
                          api={api}
                          datasetId={datasetId}
                          versionId={v.id}
                          otherVersions={versions.filter((o) => o.id !== v.id)}
                          fieldNames={fieldNames}
                        />
                      ) : null}
                      {open === "annotate" ? (
                        <AnnotationsPanel api={api} datasetId={datasetId} versionId={v.id} />
                      ) : null}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {perturbations.length > 0 ? (
        <div>
          <h2 className="mb-3 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Perturbation runs
          </h2>
          <ul className="flex flex-col gap-2" aria-label="Perturbation runs">
            {perturbations.map((j) => (
              <li key={j.id}>
                <Link
                  href={`/app/perturbations/${j.id}`}
                  className="flex items-center justify-between gap-3 rounded-xl border border-border bg-card px-4 py-3 text-sm transition-colors hover:border-terracotta/40 hover:bg-muted/40"
                >
                  <span className="font-medium capitalize">
                    {String((j.spec as { family?: string })?.family ?? "perturbation").replace("_", " ")}
                  </span>
                  <span className="font-[family-name:var(--font-data)] text-xs tracking-wide text-muted-foreground uppercase">
                    {j.status}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Loads a graph version's node-link artifact and hands it to the
 * {@link GraphExplorer}. The artifact fetch is best-effort — a version that
 * is not a graph (or a route that has not landed) surfaces a friendly note
 * rather than an error.
 *
 * Guard rail: a version whose element count (`row_count` = nodes + edges)
 * exceeds {@link GRAPH_VISUALIZE_MAX_ELEMENTS} is *never* fetched — parsing a
 * huge node-link JSON client-side would OOM the tab. Instead we show the
 * aggregate size and point the user at the download.
 */
function VisualizePanel({
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
  const [artifact, setArtifact] = useState<GraphArtifact | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState<string | null>(null);
  const tooLarge = rowCount > GRAPH_VISUALIZE_MAX_ELEMENTS;

  useEffect(() => {
    if (tooLarge) return;
    let cancelled = false;
    api.fetchGraphArtifact(datasetId, versionId).then(
      (data) => {
        if (cancelled) return;
        if (data && Array.isArray(data.nodes)) {
          setArtifact(data);
          setState("ready");
        } else {
          setMessage("This version does not contain a graph artifact.");
          setState("error");
        }
      },
      (err) => {
        if (cancelled) return;
        setMessage(
          err instanceof Error ? err.message : "Could not load the graph for this version.",
        );
        setState("error");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, datasetId, versionId, tooLarge]);

  if (tooLarge)
    return (
      <div className="flex flex-col gap-2 text-sm">
        <p className="font-medium text-foreground">
          Graph too large to visualize in-browser ({rowCount.toLocaleString()} elements).
        </p>
        <p className="text-muted-foreground">
          Rendering this many nodes and edges in the browser is not supported — download the
          version to inspect it in a desktop graph tool.
        </p>
      </div>
    );

  if (state === "loading") return <Loading label="Loading graph…" />;
  if (state === "error" || !artifact)
    return (
      <p className="text-sm text-muted-foreground">
        {message ?? "No graph to display."}
      </p>
    );
  return <GraphExplorer graph={artifact} />;
}

function ActionButton({
  active,
  onClick,
  icon: Icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof FileDown;
  children: React.ReactNode;
}) {
  return (
    <Button type="button" variant={active ? "secondary" : "outline"} size="sm" aria-expanded={active} onClick={onClick}>
      <Icon className="size-3.5" data-icon="inline-start" />
      {children}
    </Button>
  );
}
