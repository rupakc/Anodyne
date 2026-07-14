// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

import { EvaluatePanel } from "@/app/app/datasets/[id]/evaluate-panel";
import { ApiError, type EvaluationRun, type TaskMetricsCatalog } from "@/lib/api";
import { baseMockApi } from "./mock-api";

beforeEach(() => pushMock.mockClear());

const CATALOG: TaskMetricsCatalog = {
  task_type: "text_classification",
  available_metrics: [
    { key: "accuracy", label: "Accuracy", description: "Exact-match label accuracy.", requires_llm: false },
    { key: "macro_f1", label: "Macro F1", description: "F1 averaged across classes.", requires_llm: false },
  ],
};

const RUN: EvaluationRun = {
  id: "eval-1",
  tenant_id: "t-1",
  dataset_id: "d-1",
  dataset_version_id: "v-1",
  status: "pending",
  progress: 0,
  message: "",
  config: {},
  created_at: "2026-07-12T00:00:00Z",
};

describe("EvaluatePanel — standard metrics", () => {
  it("fetches the task-metrics catalog and renders all metrics checked by default, humanizing the task type", async () => {
    const api = baseMockApi({ fetchTaskMetrics: vi.fn().mockResolvedValue(CATALOG) });

    render(<EvaluatePanel api={api} datasetId="d-1" versionId="v-1" otherVersions={[]} fieldNames={[]} />);

    await waitFor(() => expect(api.fetchTaskMetrics).toHaveBeenCalledWith("d-1", "v-1"));
    expect(await screen.findByText("Text classification")).toBeInTheDocument();

    const accuracyBox = await screen.findByRole("checkbox", { name: /accuracy/i });
    const f1Box = screen.getByRole("checkbox", { name: /macro f1/i });
    expect(accuracyBox).toBeChecked();
    expect(f1Box).toBeChecked();
  });

  it("excludes an unchecked metric's key from the submitted selected_metrics", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      fetchTaskMetrics: vi.fn().mockResolvedValue(CATALOG),
      evaluate: vi.fn().mockResolvedValue(RUN),
    });

    render(<EvaluatePanel api={api} datasetId="d-1" versionId="v-1" otherVersions={[]} fieldNames={[]} />);

    const f1Box = await screen.findByRole("checkbox", { name: /macro f1/i });
    await user.click(f1Box);
    expect(f1Box).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /launch evaluation/i }));

    await waitFor(() => expect(api.evaluate).toHaveBeenCalled());
    const [, , input] = (api.evaluate as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(input.selected_metrics).toEqual(["accuracy"]);
  });

  it("submits every fetched metric key when nothing is unchecked", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      fetchTaskMetrics: vi.fn().mockResolvedValue(CATALOG),
      evaluate: vi.fn().mockResolvedValue(RUN),
    });

    render(<EvaluatePanel api={api} datasetId="d-1" versionId="v-1" otherVersions={[]} fieldNames={[]} />);
    await screen.findByText("Text classification");

    await user.click(screen.getByRole("button", { name: /launch evaluation/i }));

    await waitFor(() => expect(api.evaluate).toHaveBeenCalled());
    const [, , input] = (api.evaluate as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(input.selected_metrics).toEqual(["accuracy", "macro_f1"]);
  });

  it("refetches the task-metrics catalog with the target field once one is selected", async () => {
    const user = userEvent.setup();
    const TABULAR_CATALOG: TaskMetricsCatalog = {
      task_type: "tabular_classification",
      available_metrics: [
        { key: "f1", label: "F1", description: "F1 score.", requires_llm: false },
      ],
    };
    const fetchTaskMetrics = vi
      .fn()
      .mockResolvedValueOnce(CATALOG)
      .mockResolvedValueOnce(TABULAR_CATALOG);
    const api = baseMockApi({ fetchTaskMetrics });

    render(
      <EvaluatePanel
        api={api}
        datasetId="d-1"
        versionId="v-1"
        otherVersions={[]}
        fieldNames={["churned"]}
      />,
    );

    // Initial fetch (no target selected yet) is called with just the 2 args.
    await waitFor(() => expect(fetchTaskMetrics).toHaveBeenNthCalledWith(1, "d-1", "v-1"));
    await screen.findByText("Text classification");

    await user.selectOptions(screen.getByLabelText(/target field/i), "churned");

    await waitFor(() =>
      expect(fetchTaskMetrics).toHaveBeenNthCalledWith(2, "d-1", "v-1", "churned"),
    );
    expect(await screen.findByText("Tabular classification")).toBeInTheDocument();
  });

  it("hides the panel gracefully when available_metrics is empty", async () => {
    const api = baseMockApi({
      fetchTaskMetrics: vi.fn().mockResolvedValue({ task_type: "generic", available_metrics: [] }),
    });

    render(<EvaluatePanel api={api} datasetId="d-1" versionId="v-1" otherVersions={[]} fieldNames={[]} />);

    await waitFor(() => expect(api.fetchTaskMetrics).toHaveBeenCalled());
    expect(screen.queryByText("Standard metrics")).not.toBeInTheDocument();
  });

  it("hides the panel gracefully when the task-metrics fetch fails, without blocking evaluation", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      fetchTaskMetrics: vi.fn().mockRejectedValue(new ApiError(404, "not found")),
      evaluate: vi.fn().mockResolvedValue(RUN),
    });

    render(<EvaluatePanel api={api} datasetId="d-1" versionId="v-1" otherVersions={[]} fieldNames={[]} />);

    await waitFor(() => expect(api.fetchTaskMetrics).toHaveBeenCalled());
    expect(screen.queryByText("Standard metrics")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /launch evaluation/i }));
    await waitFor(() => expect(api.evaluate).toHaveBeenCalled());
    const [, , input] = (api.evaluate as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(input.selected_metrics).toBeUndefined();
  });
});
