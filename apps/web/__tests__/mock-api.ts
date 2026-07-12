import { vi } from "vitest";
import type { ApiClient } from "@/lib/api";

/**
 * A fully-stubbed {@link ApiClient} where every method is an unconfigured
 * `vi.fn()`. Tests spread this in and then override just the handful of
 * methods they exercise — so adding a new method to the `ApiClient` contract
 * never breaks unrelated test files.
 */
export function baseMockApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    createDataset: vi.fn(),
    listDatasets: vi.fn(),
    getDataset: vi.fn(),
    updateDataset: vi.fn(),
    generate: vi.fn(),
    getJob: vi.fn(),
    listVersions: vi.fn(),
    downloadUrl: vi.fn(),
    listTemplates: vi.fn(),
    createFromTemplate: vi.fn(),
    listProviders: vi.fn(),
    registerProvider: vi.fn(),
    deleteProvider: vi.fn(),
    uploadSample: vi.fn(),
    createAudioDataset: vi.fn(),
    createImageDataset: vi.fn(),
    exportVersion: vi.fn(),
    perturb: vi.fn(),
    getPerturbationJob: vi.fn(),
    listPerturbationJobs: vi.fn(),
    evaluate: vi.fn(),
    getEvaluation: vi.fn(),
    getEvaluationReport: vi.fn(),
    evaluationReportUrl: vi.fn(),
    listReviews: vi.fn(),
    getReview: vi.fn(),
    submitReviewDecision: vi.fn(),
    listAnnotations: vi.fn(),
    createAnnotation: vi.fn(),
    deleteAnnotation: vi.fn(),
    submitFeedback: vi.fn(),
    ...overrides,
  };
}
