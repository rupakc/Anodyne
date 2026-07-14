/**
 * Typed client for the api-gateway's dataset lifecycle endpoints
 * (`packages/anodyne-dataset/src/anodyne_dataset/models.py` is the source
 * of truth these types mirror). Every request attaches
 * `Authorization: Bearer <accessToken>` using the OIDC access token Task 11
 * puts on `session.accessToken` (see `apps/web/auth.ts`).
 *
 * The base URL comes from `NEXT_PUBLIC_API_BASE` (a public, non-secret
 * build-time value — see docs/dev-runbook.md), defaulting to the local
 * gateway at `http://localhost:8000`.
 */

export type Modality = "tabular" | "text" | "image" | "audio" | "video" | "graph";

export const SEMANTIC_TYPES = [
  "integer",
  "float",
  "boolean",
  "categorical",
  "datetime",
  "name",
  "email",
  "address",
  "text",
] as const;

export type SemanticType = (typeof SEMANTIC_TYPES)[number];

export const JOB_STATUSES = [
  "pending",
  "running",
  "awaiting_review",
  "succeeded",
  "failed",
] as const;

export type JobStatus = (typeof JOB_STATUSES)[number];

export interface FieldSpec {
  name: string;
  semantic_type: SemanticType;
  nullable: boolean;
  constraints: Record<string, unknown>;
  distribution: string | null;
}

export interface DatasetSpec {
  id: string;
  tenant_id: string;
  name: string;
  description: string;
  modality: Modality;
  source: string;
  fields: FieldSpec[];
  target_rows: number;
  directives: Record<string, unknown>;
  status: string;
  created_at: string;
}

export interface GenerationJob {
  id: string;
  tenant_id: string;
  dataset_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  workflow_id: string | null;
}

export interface DatasetVersion {
  id: string;
  tenant_id: string;
  dataset_id: string;
  artifact_uri: string;
  format: string;
  row_count: number;
  checksum: string;
  /** Present on derived versions (e.g. produced by a perturbation run). */
  parent_version_id?: string | null;
  created_at: string;
}

export type DatasetSource = "description" | "sample";

export interface CreateDatasetInput {
  name: string;
  description: string;
  target_rows: number;
  /** `POST /datasets` also accepts `source`/`modality`; defaults keep the classic tabular-from-description flow. */
  source?: DatasetSource;
  modality?: Modality;
  /** Free-form generation directives (topology, ontology, community structure, …). */
  directives?: Record<string, unknown>;
}

/**
 * A starter template blueprint (`GET /templates`) mirroring
 * `packages/anodyne-templates/src/anodyne_templates/models.py`'s `DatasetTemplate`.
 */
export interface DatasetTemplate {
  key: string;
  name: string;
  description: string;
  category: string;
  modality: Modality;
  fields: FieldSpec[];
  default_target_rows: number;
  default_directives: Record<string, unknown>;
}

export interface CreateFromTemplateInput {
  template_key: string;
  name?: string;
  target_rows?: number;
  directives?: Record<string, unknown>;
}

export interface UpdateDatasetInput {
  name?: string;
  target_rows?: number;
  fields?: FieldSpec[];
  /** For the graph modality: edit the proposed ontology (and other directives) before generating. */
  directives?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Graph modality (spec docs/superpowers/specs/2026-07-13-graph-modality-design.md).
// The canonical model is a typed property graph. A graph dataset is created via
// `POST /datasets` with `modality:"graph"`; the gateway returns a proposed
// ontology in `directives.ontology`. Versions serialize a node-link JSON
// artifact ({ontology, nodes, edges, metrics}) which the explorer fetches.
// ---------------------------------------------------------------------------

export type GraphSource = "description" | "sample" | "ontology";

/** One datatype property on a node type (or edge type). */
export interface GraphPropertySpec {
  name: string;
  datatype: string;
}

export interface GraphNodeType {
  name: string;
  properties: GraphPropertySpec[];
  /** Subclass hierarchy (T-Box `subClassOf`), when the ontology defines one. */
  subclass_of?: string | null;
}

export interface GraphEdgeType {
  name: string;
  source_type: string;
  target_type: string;
  properties?: GraphPropertySpec[];
}

export interface GraphOntology {
  node_types: GraphNodeType[];
  edge_types: GraphEdgeType[];
}

export interface GraphNode {
  id: string;
  type: string;
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  type: string;
  source: string;
  target: string;
  properties?: Record<string, unknown>;
}

/** Full-graph aggregate stats emitted alongside the artifact. */
export interface GraphMetrics {
  node_count?: number;
  edge_count?: number;
  nodes_by_type?: Record<string, number>;
  edges_by_type?: Record<string, number>;
  density?: number;
  [key: string]: unknown;
}

/** The node-link JSON artifact a graph version serializes. */
export interface GraphArtifact {
  ontology: GraphOntology;
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics?: GraphMetrics;
}

export interface CreateGraphDatasetInput {
  name: string;
  description: string;
  /** Maps to node count (shared `target_rows`). */
  target_rows: number;
  source?: GraphSource;
  directives?: Record<string, unknown>;
}

/** Topology models the generator supports (directive `topology.model`). */
export const GRAPH_TOPOLOGIES = [
  { value: "barabasi_albert", label: "Scale-free (Barabási–Albert)", hint: "Power-law degree; a few hubs." },
  { value: "watts_strogatz", label: "Small-world (Watts–Strogatz)", hint: "High clustering, short paths." },
  { value: "stochastic_block", label: "Communities (Stochastic Block Model)", hint: "Dense within groups." },
  { value: "ontology_constrained", label: "Ontology-constrained", hint: "Edges only between compatible types." },
] as const;

export type GraphTopology = (typeof GRAPH_TOPOLOGIES)[number]["value"];

/** Graph export projections offered on a graph version (GC sub-system). */
export const GRAPH_EXPORT_FORMATS = [
  { value: "node-link", label: "Node-link JSON" },
  { value: "turtle", label: "Turtle (RDF)" },
  { value: "jsonld", label: "JSON-LD" },
  { value: "graphml", label: "GraphML" },
  { value: "cypher", label: "Cypher script" },
  { value: "neo4j-csv", label: "Neo4j admin CSV" },
  { value: "owl", label: "OWL (ontology)" },
] as const;

export type GraphExportFormat = (typeof GRAPH_EXPORT_FORMATS)[number]["value"];

export interface GenerateOptions {
  seed?: number;
  /** Optional provider override (`model_config_id`) for LLM-backed modalities. */
  model_config_id?: string;
  /**
   * Opt into a human-review gate before generation runs. When true the gateway
   * parks the workflow and creates a pending review task (see `/app/reviews`);
   * defaults to false (auto-approve, the historical behavior).
   */
  require_review?: boolean;
}

// ---------------------------------------------------------------------------
// Sub-system H additions: providers, extra modalities, export, perturbation,
// evaluation, and the (in-flight) HITL review/annotation/feedback contract.
// Field names mirror the gateway route modules exactly; see the design note
// docs/superpowers/specs/2026-07-12-web-ui-h-design.md for the endpoint map.
// ---------------------------------------------------------------------------

/** The four tenant-scoped provider registries the gateway exposes, by URL segment. */
export type ProviderKind = "models" | "image-providers" | "video-providers" | "audio-providers";

export const PROVIDER_KINDS: { kind: ProviderKind; label: string; modality: Modality | "llm" }[] = [
  { kind: "models", label: "LLM / Text", modality: "llm" },
  { kind: "image-providers", label: "Image", modality: "image" },
  { kind: "audio-providers", label: "Audio", modality: "audio" },
  { kind: "video-providers", label: "Video", modality: "video" },
];

/** `ModelConfig` as returned by every provider registry (secret_ref stripped server-side). */
export interface ProviderConfig {
  id: string;
  tenant_id: string;
  name: string;
  provider: string;
  model: string;
  params: Record<string, unknown>;
  api_base: string | null;
  enabled: boolean;
}

export interface RegisterProviderInput {
  name: string;
  provider: string;
  model: string;
  api_key?: string | null;
  api_base?: string | null;
  params?: Record<string, unknown>;
}

/** One column of a profiled sample (`POST /datasets/{id}/sample`). */
export interface ColumnProfile {
  name: string;
  semantic_type: string;
  nullable: boolean;
  null_rate: number;
  distinct_count?: number | null;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  categories?: unknown[] | null;
}

export interface SampleProfile {
  id: string;
  tenant_id: string;
  dataset_id: string;
  row_count: number;
  columns: ColumnProfile[];
  correlations: Record<string, unknown>;
  sample_uri: string;
  sample_filename: string;
  created_at: string;
}

export interface SampleUploadResult {
  dataset: DatasetSpec;
  profile: SampleProfile;
}

export interface CreateAudioDatasetInput {
  name: string;
  description?: string;
  target_rows: number;
  directives?: Record<string, unknown>;
}

export interface CreateImageDatasetInput {
  name: string;
  description?: string;
  target_count: number;
  labels?: string[];
  directives?: Record<string, unknown>;
}

export type ExportFormat = "csv" | "json" | "parquet" | "arrow";

export const EXPORT_FORMATS: ExportFormat[] = ["csv", "json", "parquet", "arrow"];

export interface ExportArtifact {
  id: string;
  tenant_id: string;
  dataset_id: string;
  version_id: string;
  format: string;
  row_count: number;
  object_key: string;
  created_at: string;
}

export const PERTURBATION_FAMILIES = ["noise", "drift", "outliers", "bias", "edge_case"] as const;
export type PerturbationFamily = (typeof PERTURBATION_FAMILIES)[number];

export interface PerturbInput {
  family: PerturbationFamily;
  intensity?: number;
  target_fields?: string[];
  params?: Record<string, unknown>;
  seed?: number;
}

export interface PerturbationJob {
  id: string;
  tenant_id: string;
  dataset_id: string;
  parent_version_id: string;
  spec: Record<string, unknown>;
  status: JobStatus;
  progress: number;
  message: string;
  workflow_id?: string | null;
  result_version_id?: string | null;
  created_at: string;
}

export const EVAL_DIMENSIONS = [
  "fidelity",
  "diversity",
  "privacy",
  "utility",
  "bias",
  "qualitative",
] as const;
export type EvalDimension = (typeof EVAL_DIMENSIONS)[number];

export interface EvaluateInput {
  reference_version_id?: string;
  seed?: number;
  sensitive_field?: string;
  target_field?: string;
  text_column?: string;
  model_config_id?: string;
  sample_rows?: number;
  weights?: Record<string, number>;
  /** Keys from `TaskMetricsCatalog.available_metrics` to include; omit to run the full default set. */
  selected_metrics?: string[];
}

/** One standard metric offered for a dataset version's detected task class (`GET .../task-metrics`). */
export interface MetricSpec {
  key: string;
  label: string;
  description: string;
  requires_llm: boolean;
}

/** The per-task standard-metrics catalog for a dataset version. */
export interface TaskMetricsCatalog {
  task_type: string;
  available_metrics: MetricSpec[];
}

export interface EvaluationRun {
  id: string;
  tenant_id: string;
  dataset_id: string;
  dataset_version_id: string;
  reference_version_id?: string | null;
  status: JobStatus;
  progress: number;
  message: string;
  workflow_id?: string | null;
  report_uri?: string | null;
  report_html_uri?: string | null;
  overall_score?: number | null;
  config: Record<string, unknown>;
  created_at: string;
}

export interface ExpertScore {
  dimension: string;
  score: number;
  rationale: string;
  metrics: Record<string, unknown>;
  recommendations: string[];
}

export interface EvaluationReport {
  id: string;
  tenant_id: string;
  dataset_id: string;
  dataset_version_id: string;
  reference_version_id?: string | null;
  overall_score: number;
  expert_scores: ExpertScore[];
  weights: Record<string, number>;
  recommendations: string[];
  summary: string;
  created_at: string;
}

// --- HITL contract (routes built in parallel; callers tolerate 404/501) -----

export type ReviewDecision = "approve" | "reject" | "changes_requested";

export interface ReviewItem {
  id: string;
  tenant_id?: string;
  status: string;
  kind?: string;
  target_type?: string;
  target_id?: string;
  title?: string;
  summary?: string;
  payload?: Record<string, unknown>;
  created_at?: string;
}

export interface ReviewDecisionInput {
  decision: ReviewDecision;
  comment?: string;
}

export interface Annotation {
  id: string;
  tenant_id?: string;
  dataset_id?: string;
  version_id?: string;
  row_index?: number | null;
  record_id?: string | null;
  label?: string | null;
  tags?: string[];
  comment?: string | null;
  created_at?: string;
}

export interface CreateAnnotationInput {
  row_index?: number;
  record_id?: string;
  label?: string;
  tags?: string[];
  comment?: string;
}

export interface FeedbackInput {
  target_type: string;
  target_id: string;
  rating?: number;
  thumbs?: "up" | "down";
  comment?: string;
  expert_override?: Record<string, unknown>;
}

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** Message shown when a credential submission is blocked for being cleartext. */
export const HTTPS_REQUIRED_MESSAGE =
  "HTTPS required to submit credentials. Configure the gateway (NEXT_PUBLIC_API_BASE) to use https, or connect over localhost for local development.";

/**
 * Whether submitting a secret (e.g. a provider `api_key`) to `baseUrl` would
 * send it in cleartext. `http://` is only safe to a loopback host
 * (localhost / 127.0.0.1 / [::1]) for local dev; any other `http://` host must
 * be blocked so credentials never traverse the network unencrypted. `https`
 * (and a malformed/relative base, which stays same-origin) is always allowed.
 */
export function isCredentialSubmissionInsecure(baseUrl: string = API_BASE): boolean {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    return false;
  }
  if (url.protocol !== "http:") return false;
  const host = url.hostname.toLowerCase();
  const loopback = host === "localhost" || host === "127.0.0.1" || host === "::1" || host === "[::1]";
  return !loopback;
}

/**
 * Base URL for the live job-progress socket (`WS /jobs/{id}/stream`). Reads
 * `NEXT_PUBLIC_WS_BASE` (public, non-secret — see docs/dev-runbook.md) and
 * otherwise derives it from {@link API_BASE} by swapping the `http(s)`
 * scheme for `ws(s)`, so a single env var change (`NEXT_PUBLIC_API_BASE`)
 * still works out of the box in the common case of gateway and socket
 * sharing a host.
 */
export const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE ?? API_BASE.replace(/^http/, "ws");

/** Builds the `WS /jobs/{id}/stream` URL for {@link WS_BASE}. */
export function jobStreamUrl(jobId: string, baseUrl: string = WS_BASE): string {
  return `${baseUrl}/jobs/${jobId}/stream`;
}

/**
 * Extracts the filename from a `Content-Disposition` response header, e.g.
 * `attachment; filename="customers.csv"`. Prefers the RFC 5987 extended form
 * (`filename*=UTF-8''...`) when present; falls back to the plain
 * `filename="..."` form. Returns `null` if the header is absent or has
 * neither form (the caller supplies a fallback name in that case).
 */
export function filenameFromContentDisposition(header: string | null): string | null {
  if (!header) return null;
  const extended = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(header);
  if (extended?.[1]) {
    try {
      return decodeURIComponent(extended[1].trim().replace(/^"|"$/g, ""));
    } catch {
      // Malformed percent-encoding: fall through to the plain form below.
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(header);
  return plain?.[1]?.trim() || null;
}

/** Thrown for any non-2xx gateway response; carries the HTTP status for callers that care. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface ApiClient {
  createDataset(input: CreateDatasetInput): Promise<DatasetSpec>;
  listDatasets(): Promise<DatasetSpec[]>;
  getDataset(id: string): Promise<DatasetSpec>;
  updateDataset(id: string, patch: UpdateDatasetInput): Promise<DatasetSpec>;
  generate(id: string, opts?: GenerateOptions): Promise<GenerationJob>;
  getJob(id: string): Promise<GenerationJob>;
  listVersions(id: string): Promise<DatasetVersion[]>;
  /**
   * Streams one version's artifact bytes through the gateway and triggers a
   * browser download. Deliberately NOT a presigned URL: those can go stale
   * ("Signature has expired") if the page stays open, the laptop sleeps, or
   * the clock jumps between load and click. Resolves to the filename used.
   */
  downloadVersionFile(datasetId: string, versionId: string): Promise<string>;
  /** Lists the starter template catalog (`GET /templates`). */
  listTemplates(): Promise<DatasetTemplate[]>;
  /** Builds and persists a `DatasetSpec` from a catalog template (`POST /datasets/from-template`). */
  createFromTemplate(input: CreateFromTemplateInput): Promise<DatasetSpec>;

  // --- providers (per-modality registries) ---
  listProviders(kind: ProviderKind): Promise<ProviderConfig[]>;
  registerProvider(kind: ProviderKind, input: RegisterProviderInput): Promise<ProviderConfig>;
  deleteProvider(kind: ProviderKind, id: string): Promise<void>;

  // --- extra generation modalities ---
  /** Uploads a sample file (multipart) and returns the created dataset + its profile. */
  uploadSample(datasetId: string, file: File): Promise<SampleUploadResult>;
  createAudioDataset(input: CreateAudioDatasetInput): Promise<DatasetSpec>;
  createImageDataset(input: CreateImageDatasetInput): Promise<DatasetSpec>;

  // --- graph modality ---
  /** Creates a graph dataset (`POST /datasets`, `modality:"graph"`); the gateway proposes an ontology in `directives.ontology`. */
  createGraphDataset(input: CreateGraphDatasetInput): Promise<DatasetSpec>;
  /** Fetches + parses a graph version's node-link JSON artifact for in-browser visualization. */
  fetchGraphArtifact(datasetId: string, versionId: string): Promise<GraphArtifact>;
  /** Exports a graph version to a graph format via the streaming download; resolves to the filename used. */
  exportGraphVersion(datasetId: string, versionId: string, format: GraphExportFormat): Promise<string>;

  // --- export ---
  /**
   * Creates the export artifact server-side, then streams its bytes through
   * the gateway and triggers a browser download (no presigned URL — see
   * `downloadVersionFile`). Resolves to the filename used.
   */
  exportVersion(datasetId: string, versionId: string, format?: ExportFormat): Promise<string>;

  // --- perturbation ---
  perturb(datasetId: string, versionId: string, input: PerturbInput): Promise<PerturbationJob>;
  getPerturbationJob(id: string): Promise<PerturbationJob>;
  listPerturbationJobs(datasetId: string): Promise<PerturbationJob[]>;

  // --- evaluation ---
  /**
   * Fetches the detected task class + standard-metric catalog for a version
   * (`GET .../task-metrics`). Pass `targetField` when the caller has one
   * selected (e.g. before launching a tabular evaluation) so the catalog
   * matches the task class the evaluation workflow will actually resolve
   * (`tabular_classification`/`regression` instead of the target-blind
   * `generic` fallback).
   */
  fetchTaskMetrics(
    datasetId: string,
    versionId: string,
    targetField?: string,
  ): Promise<TaskMetricsCatalog>;
  evaluate(datasetId: string, versionId: string, input: EvaluateInput): Promise<EvaluationRun>;
  getEvaluation(id: string): Promise<EvaluationRun>;
  getEvaluationReport(id: string): Promise<EvaluationReport>;
  /**
   * Streams the evaluation report through the gateway and triggers a browser
   * download (no presigned URL — see `downloadVersionFile`). Resolves to the
   * filename used.
   */
  downloadReport(id: string, format?: "html" | "json"): Promise<string>;

  // --- HITL: reviews / annotations / feedback ---
  listReviews(status?: string): Promise<ReviewItem[]>;
  getReview(id: string): Promise<ReviewItem>;
  submitReviewDecision(id: string, input: ReviewDecisionInput): Promise<ReviewItem>;
  listAnnotations(datasetId: string, versionId: string): Promise<Annotation[]>;
  createAnnotation(
    datasetId: string,
    versionId: string,
    input: CreateAnnotationInput,
  ): Promise<Annotation>;
  deleteAnnotation(id: string): Promise<void>;
  submitFeedback(input: FeedbackInput): Promise<void>;
}

/**
 * Builds an {@link ApiClient} bound to a single access token. Pass
 * `session?.accessToken` from a signed-in request; when it's undefined the
 * client still works (useful for `/healthz`-style anonymous calls) but any
 * RBAC-guarded route will 401.
 */
export function createApiClient(accessToken: string | undefined, baseUrl: string = API_BASE): ApiClient {
  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const headers = new Headers(init?.headers);
    headers.set("Content-Type", "application/json");
    if (accessToken) {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }

    const res = await fetch(`${baseUrl}${path}`, { ...init, headers });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new ApiError(
        res.status,
        detail || `${init?.method ?? "GET"} ${path} failed with ${res.status}`,
      );
    }
    if (res.status === 204) {
      return undefined as T;
    }
    return (await res.json()) as T;
  }

  /** Multipart variant: never sets `Content-Type` so the browser adds the boundary. */
  async function requestForm<T>(path: string, form: FormData): Promise<T> {
    const headers = new Headers();
    if (accessToken) {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    const res = await fetch(`${baseUrl}${path}`, { method: "POST", body: form, headers });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new ApiError(res.status, detail || `POST ${path} failed with ${res.status}`);
    }
    return (await res.json()) as T;
  }

  /**
   * Authenticated fetch that streams a binary response straight into a
   * browser download, instead of the gateway handing back a presigned
   * MinIO/S3 URL (which can go stale — "Signature has expired" — if the page
   * stays open, the laptop sleeps, or the clock jumps between load and
   * click). Derives the filename from `Content-Disposition`, falling back to
   * `opts.fallbackName`. Resolves to the filename actually used.
   */
  async function downloadToFile(
    path: string,
    opts?: { method?: string; body?: unknown; fallbackName?: string },
  ): Promise<string> {
    const headers = new Headers();
    if (accessToken) {
      headers.set("Authorization", `Bearer ${accessToken}`);
    }
    if (opts?.body !== undefined) {
      headers.set("Content-Type", "application/json");
    }

    const res = await fetch(`${baseUrl}${path}`, {
      method: opts?.method ?? "GET",
      headers,
      body: opts?.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new ApiError(
        res.status,
        detail || `${opts?.method ?? "GET"} ${path} failed with ${res.status}`,
      );
    }

    const blob = await res.blob();
    const filename =
      filenameFromContentDisposition(res.headers.get("Content-Disposition")) ??
      opts?.fallbackName ??
      "download";

    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);

    return filename;
  }

  return {
    createDataset: (input) =>
      request<DatasetSpec>("/datasets", { method: "POST", body: JSON.stringify(input) }),
    listDatasets: () => request<DatasetSpec[]>("/datasets"),
    getDataset: (id) => request<DatasetSpec>(`/datasets/${id}`),
    updateDataset: (id, patch) =>
      request<DatasetSpec>(`/datasets/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
    generate: (id, opts) =>
      request<GenerationJob>(`/datasets/${id}/generate`, {
        method: "POST",
        body: JSON.stringify({
          seed: opts?.seed ?? 0,
          ...(opts?.require_review ? { require_review: true } : {}),
        }),
      }),
    getJob: (id) => request<GenerationJob>(`/jobs/${id}`),
    listVersions: (id) => request<DatasetVersion[]>(`/datasets/${id}/versions`),
    downloadVersionFile: (datasetId, versionId) =>
      downloadToFile(`/datasets/${datasetId}/versions/${versionId}/download`, {
        fallbackName: `dataset-${datasetId}-version-${versionId}`,
      }),
    listTemplates: () => request<DatasetTemplate[]>("/templates"),
    createFromTemplate: (input) =>
      request<DatasetSpec>("/datasets/from-template", {
        method: "POST",
        body: JSON.stringify(input),
      }),

    listProviders: (kind) => request<ProviderConfig[]>(`/${kind}`),
    registerProvider: (kind, input) =>
      request<ProviderConfig>(`/${kind}`, { method: "POST", body: JSON.stringify(input) }),
    deleteProvider: (kind, id) =>
      request<void>(`/${kind}/${id}`, { method: "DELETE" }),

    uploadSample: (datasetId, file) => {
      const form = new FormData();
      form.append("file", file);
      return requestForm<SampleUploadResult>(`/datasets/${datasetId}/sample`, form);
    },
    createAudioDataset: (input) =>
      request<DatasetSpec>("/datasets/audio", { method: "POST", body: JSON.stringify(input) }),
    createImageDataset: (input) =>
      request<DatasetSpec>("/datasets/image", { method: "POST", body: JSON.stringify(input) }),

    createGraphDataset: (input) =>
      request<DatasetSpec>("/datasets", {
        method: "POST",
        body: JSON.stringify({ ...input, modality: "graph" }),
      }),
    fetchGraphArtifact: (datasetId, versionId) =>
      request<GraphArtifact>(`/datasets/${datasetId}/versions/${versionId}/download`),
    exportGraphVersion: (datasetId, versionId, format) =>
      downloadToFile(
        `/datasets/${datasetId}/versions/${versionId}/export?format=${encodeURIComponent(format)}`,
        { method: "POST", fallbackName: `graph-${datasetId}-${versionId}.${format}` },
      ),

    exportVersion: (datasetId, versionId, format) =>
      downloadToFile(`/datasets/${datasetId}/versions/${versionId}/export`, {
        method: "POST",
        body: format ? { format } : {},
        fallbackName: `export-${datasetId}-${versionId}`,
      }),

    perturb: (datasetId, versionId, input) =>
      request<PerturbationJob>(`/datasets/${datasetId}/versions/${versionId}/perturb`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    getPerturbationJob: (id) => request<PerturbationJob>(`/perturbation-jobs/${id}`),
    listPerturbationJobs: (datasetId) =>
      request<PerturbationJob[]>(`/datasets/${datasetId}/perturbation-jobs`),

    fetchTaskMetrics: (datasetId, versionId, targetField) =>
      request<TaskMetricsCatalog>(
        `/datasets/${datasetId}/versions/${versionId}/task-metrics${
          targetField ? `?target_field=${encodeURIComponent(targetField)}` : ""
        }`,
      ),
    evaluate: (datasetId, versionId, input) =>
      request<EvaluationRun>(`/datasets/${datasetId}/versions/${versionId}/evaluate`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    getEvaluation: (id) => request<EvaluationRun>(`/evaluations/${id}`),
    getEvaluationReport: (id) => request<EvaluationReport>(`/evaluations/${id}/report`),
    downloadReport: (id, format = "html") =>
      downloadToFile(`/evaluations/${id}/report/download?format=${encodeURIComponent(format)}`, {
        fallbackName: `evaluation-${id}.${format}`,
      }),

    listReviews: (status) =>
      request<ReviewItem[]>(`/reviews${status ? `?status=${encodeURIComponent(status)}` : ""}`),
    getReview: (id) => request<ReviewItem>(`/reviews/${id}`),
    submitReviewDecision: (id, input) =>
      request<ReviewItem>(`/reviews/${id}/decision`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    listAnnotations: (datasetId, versionId) =>
      request<Annotation[]>(`/datasets/${datasetId}/versions/${versionId}/annotations`),
    createAnnotation: (datasetId, versionId, input) =>
      request<Annotation>(`/datasets/${datasetId}/versions/${versionId}/annotations`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    deleteAnnotation: (id) => request<void>(`/annotations/${id}`, { method: "DELETE" }),
    submitFeedback: (input) =>
      request<void>("/feedback", { method: "POST", body: JSON.stringify(input) }),
  };
}
