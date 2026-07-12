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

export type Modality = "tabular" | "text" | "image" | "audio" | "video";

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
}

export interface GenerateOptions {
  seed?: number;
  /** Optional provider override (`model_config_id`) for LLM-backed modalities. */
  model_config_id?: string;
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

export interface ExportResult {
  artifact: ExportArtifact;
  url: string;
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
  /** Resolves to a presigned, time-limited download URL for one version's artifact. */
  downloadUrl(datasetId: string, versionId: string): Promise<string>;
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

  // --- export ---
  exportVersion(datasetId: string, versionId: string, format?: ExportFormat): Promise<ExportResult>;

  // --- perturbation ---
  perturb(datasetId: string, versionId: string, input: PerturbInput): Promise<PerturbationJob>;
  getPerturbationJob(id: string): Promise<PerturbationJob>;
  listPerturbationJobs(datasetId: string): Promise<PerturbationJob[]>;

  // --- evaluation ---
  evaluate(datasetId: string, versionId: string, input: EvaluateInput): Promise<EvaluationRun>;
  getEvaluation(id: string): Promise<EvaluationRun>;
  getEvaluationReport(id: string): Promise<EvaluationReport>;
  evaluationReportUrl(id: string): Promise<string>;

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
        body: JSON.stringify({ seed: opts?.seed ?? 0 }),
      }),
    getJob: (id) => request<GenerationJob>(`/jobs/${id}`),
    listVersions: (id) => request<DatasetVersion[]>(`/datasets/${id}/versions`),
    downloadUrl: (datasetId, versionId) =>
      request<{ url: string }>(`/datasets/${datasetId}/versions/${versionId}/download`).then(
        (r) => r.url,
      ),
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

    exportVersion: (datasetId, versionId, format) =>
      request<ExportResult>(`/datasets/${datasetId}/versions/${versionId}/export`, {
        method: "POST",
        body: JSON.stringify(format ? { format } : {}),
      }),

    perturb: (datasetId, versionId, input) =>
      request<PerturbationJob>(`/datasets/${datasetId}/versions/${versionId}/perturb`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    getPerturbationJob: (id) => request<PerturbationJob>(`/perturbation-jobs/${id}`),
    listPerturbationJobs: (datasetId) =>
      request<PerturbationJob[]>(`/datasets/${datasetId}/perturbation-jobs`),

    evaluate: (datasetId, versionId, input) =>
      request<EvaluationRun>(`/datasets/${datasetId}/versions/${versionId}/evaluate`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    getEvaluation: (id) => request<EvaluationRun>(`/evaluations/${id}`),
    getEvaluationReport: (id) => request<EvaluationReport>(`/evaluations/${id}/report`),
    evaluationReportUrl: (id) =>
      request<{ url: string }>(`/evaluations/${id}/report/download`).then((r) => r.url),

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
