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
  created_at: string;
}

export interface CreateDatasetInput {
  name: string;
  description: string;
  target_rows: number;
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
  };
}
