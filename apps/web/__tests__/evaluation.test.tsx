// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

import { EvaluationView } from "@/app/app/evaluations/[id]/evaluation-view";
import { EvalReport } from "@/components/eval-report";
import type { EvaluationReport, EvaluationRun } from "@/lib/api";
import { baseMockApi } from "./mock-api";

const RUNNING: EvaluationRun = {
  id: "eval-1",
  tenant_id: "tenant-1",
  dataset_id: "dataset-1",
  dataset_version_id: "version-1",
  reference_version_id: null,
  status: "running",
  progress: 0.45,
  message: "Judging fidelity…",
  overall_score: null,
  config: {},
  created_at: "2026-07-12T00:00:00Z",
};

const SUCCEEDED: EvaluationRun = {
  ...RUNNING,
  status: "succeeded",
  progress: 1,
  message: "Done",
  overall_score: 0.87,
};

const REPORT: EvaluationReport = {
  id: "report-1",
  tenant_id: "tenant-1",
  dataset_id: "dataset-1",
  dataset_version_id: "version-1",
  reference_version_id: null,
  overall_score: 0.87,
  weights: {},
  summary: "The synthetic data closely matches the reference distribution.",
  recommendations: ["Increase sample diversity in the tail."],
  expert_scores: [
    { dimension: "fidelity", score: 0.9, rationale: "Marginals align well.", metrics: {}, recommendations: [] },
    { dimension: "diversity", score: 0.7, rationale: "Some mode collapse.", metrics: {}, recommendations: ["Raise temperature."] },
    { dimension: "privacy", score: 0.95, rationale: "No leakage detected.", metrics: {}, recommendations: [] },
  ],
  created_at: "2026-07-12T00:00:00Z",
};

describe("EvaluationView", () => {
  it("shows a progress bar while the run is in flight", async () => {
    const api = baseMockApi({ getEvaluation: vi.fn().mockResolvedValue(RUNNING) });

    render(<EvaluationView evaluationId="eval-1" api={api} pollIntervalMs={10_000} />);

    await waitFor(() => expect(api.getEvaluation).toHaveBeenCalledWith("eval-1"));
    expect(await screen.findByText("Evaluating")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: /evaluation progress/i })).toHaveAttribute(
      "aria-valuenow",
      "45",
    );
    // Report is not fetched until the run succeeds.
    expect(api.getEvaluationReport).not.toHaveBeenCalled();
  });

  it("renders the report once the run succeeds", async () => {
    const api = baseMockApi({
      getEvaluation: vi.fn().mockResolvedValue(SUCCEEDED),
      getEvaluationReport: vi.fn().mockResolvedValue(REPORT),
    });

    render(<EvaluationView evaluationId="eval-1" api={api} pollIntervalMs={10_000} />);

    expect(await screen.findByText("The synthetic data closely matches the reference distribution.")).toBeInTheDocument();
    expect(screen.getByText("87")).toBeInTheDocument(); // 360° score out of 100
    // Dimension names show up (radar + expert cards).
    expect(screen.getAllByText("Fidelity").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Privacy").length).toBeGreaterThan(0);
    expect(api.getEvaluationReport).toHaveBeenCalledWith("eval-1");
  });

  it("downloads the report by resolving its presigned URL and opening it", async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const api = baseMockApi({
      getEvaluation: vi.fn().mockResolvedValue(SUCCEEDED),
      getEvaluationReport: vi.fn().mockResolvedValue(REPORT),
      evaluationReportUrl: vi.fn().mockResolvedValue("https://minio.local/report-1.html?sig=abc"),
    });

    render(<EvaluationView evaluationId="eval-1" api={api} pollIntervalMs={10_000} />);

    const button = await screen.findByRole("button", { name: /download report/i });
    await user.click(button);

    await waitFor(() => expect(api.evaluationReportUrl).toHaveBeenCalledWith("eval-1"));
    await waitFor(() =>
      expect(openSpy).toHaveBeenCalledWith(
        "https://minio.local/report-1.html?sig=abc",
        "_blank",
        "noopener,noreferrer",
      ),
    );
    openSpy.mockRestore();
  });

  it("surfaces a failed run", async () => {
    const api = baseMockApi({
      getEvaluation: vi.fn().mockResolvedValue({ ...RUNNING, status: "failed", message: "judge crashed" }),
    });

    render(<EvaluationView evaluationId="eval-1" api={api} pollIntervalMs={10_000} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/judge crashed/i);
  });
});

describe("EvalReport", () => {
  it("renders the radar, per-expert cards, rationale, and recommendations", () => {
    render(<EvalReport report={REPORT} />);

    // Radar summary is exposed for screen readers.
    expect(screen.getByRole("img", { name: /dimension scores out of 100/i })).toBeInTheDocument();
    expect(screen.getByText("Marginals align well.")).toBeInTheDocument();
    expect(screen.getByText("Raise temperature.")).toBeInTheDocument();
    expect(screen.getByText("Increase sample diversity in the tail.")).toBeInTheDocument();
    // Per-expert progressbars (one per dimension).
    expect(screen.getAllByRole("progressbar").length).toBe(REPORT.expert_scores.length);
  });
});
