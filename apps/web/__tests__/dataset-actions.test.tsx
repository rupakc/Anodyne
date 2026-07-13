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

import { ExportPanel } from "@/app/app/datasets/[id]/export-panel";
import { PerturbPanel } from "@/app/app/datasets/[id]/perturb-panel";
import { AnnotationsPanel } from "@/app/app/datasets/[id]/annotations-panel";
import { FeedbackWidget } from "@/components/feedback-widget";
import { baseMockApi } from "./mock-api";
import { ApiError, type PerturbationJob } from "@/lib/api";

beforeEach(() => pushMock.mockClear());

const PJOB: PerturbationJob = {
  id: "pj-1",
  tenant_id: "t-1",
  dataset_id: "d-1",
  parent_version_id: "v-1",
  spec: { family: "noise" },
  status: "pending",
  progress: 0,
  message: "",
  created_at: "2026-07-12T00:00:00Z",
};

describe("ExportPanel", () => {
  it("shows the auto→CSV hint and streams the download through the gateway on export", async () => {
    // No presigned URL: `exportVersion` streams the artifact through an
    // authenticated fetch and triggers the browser download itself.
    const user = userEvent.setup();
    const api = baseMockApi({ exportVersion: vi.fn().mockResolvedValue("customers.csv") });

    render(<ExportPanel api={api} datasetId="d-1" versionId="v-1" rowCount={100} />);
    expect(screen.getByText(/auto picks parquet/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^export$/i }));
    await waitFor(() => expect(api.exportVersion).toHaveBeenCalledWith("d-1", "v-1", undefined));

    // No stale href is ever stored: a plain "Download" button re-runs (and re-streams) the export.
    expect(await screen.findByText(/customers\.csv/i)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /download/i })).not.toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /^download$/i })).toBeInTheDocument();
  });
});

describe("PerturbPanel", () => {
  it("launches a perturbation and navigates to its progress page", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({ perturb: vi.fn().mockResolvedValue(PJOB) });

    render(<PerturbPanel api={api} datasetId="d-1" versionId="v-1" fieldNames={["amount", "city"]} />);
    await user.click(screen.getByRole("button", { name: /^amount$/i }));
    await user.click(screen.getByRole("button", { name: /launch perturbation/i }));

    await waitFor(() => expect(api.perturb).toHaveBeenCalled());
    const [, , input] = (api.perturb as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(input).toMatchObject({ family: "noise", target_fields: ["amount"] });
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/app/perturbations/pj-1"));
  });
});

describe("AnnotationsPanel", () => {
  it("adds an annotation and lists it", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      listAnnotations: vi.fn().mockResolvedValue([]),
      createAnnotation: vi.fn().mockResolvedValue({ id: "a-1", label: "outlier", row_index: 3, tags: ["odd"] }),
    });

    render(<AnnotationsPanel api={api} datasetId="d-1" versionId="v-1" />);
    await screen.findByRole("button", { name: /add annotation/i });
    await user.type(screen.getByLabelText("Label"), "outlier");
    await user.click(screen.getByRole("button", { name: /add annotation/i }));

    await waitFor(() => expect(api.createAnnotation).toHaveBeenCalled());
    const list = await screen.findByRole("list", { name: /annotations/i });
    expect(list).toHaveTextContent("outlier");
  });

  it("degrades gracefully when the annotations route is absent (404)", async () => {
    const api = baseMockApi({
      listAnnotations: vi.fn().mockRejectedValue(new ApiError(404, "not found")),
    });
    render(<AnnotationsPanel api={api} datasetId="d-1" versionId="v-1" />);
    expect(await screen.findByText(/aren't enabled for this tenant/i)).toBeInTheDocument();
  });
});

describe("FeedbackWidget", () => {
  it("submits thumbs feedback for the target", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({ submitFeedback: vi.fn().mockResolvedValue(undefined) });

    render(<FeedbackWidget targetType="dataset" targetId="d-1" api={api} />);
    await user.click(screen.getByRole("button", { name: /thumbs up/i }));

    await waitFor(() =>
      expect(api.submitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({ target_type: "dataset", target_id: "d-1", thumbs: "up" }),
      ),
    );
    expect(await screen.findByText(/thanks/i)).toBeInTheDocument();
  });
});
