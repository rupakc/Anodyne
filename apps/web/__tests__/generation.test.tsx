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

import { Wizard } from "@/app/app/new/wizard";
import { TextWizard } from "@/app/app/new/text-wizard";
import { MediaWizard } from "@/app/app/new/media-wizard";
import { baseMockApi } from "./mock-api";
import type { DatasetSpec, GenerationJob, ProviderConfig, SampleProfile } from "@/lib/api";

const SPEC: DatasetSpec = {
  id: "ds-1",
  tenant_id: "t-1",
  name: "Sampled",
  description: "from sample",
  modality: "tabular",
  source: "sample",
  target_rows: 100,
  directives: {},
  status: "draft",
  created_at: "2026-07-12T00:00:00Z",
  fields: [{ name: "amount", semantic_type: "float", nullable: false, constraints: {}, distribution: null }],
};

const PROFILE: SampleProfile = {
  id: "p-1",
  tenant_id: "t-1",
  dataset_id: "ds-1",
  row_count: 100,
  columns: [],
  correlations: {},
  sample_uri: "s3://x",
  sample_filename: "s.csv",
  created_at: "2026-07-12T00:00:00Z",
};

const JOB: GenerationJob = {
  id: "job-9",
  tenant_id: "t-1",
  dataset_id: "ds-1",
  status: "pending",
  progress: 0,
  message: "",
  workflow_id: null,
};

const IMAGE_PROVIDER: ProviderConfig = {
  id: "prov-1",
  tenant_id: "t-1",
  name: "SDXL",
  provider: "replicate",
  model: "sdxl",
  params: {},
  api_base: null,
  enabled: true,
};

beforeEach(() => pushMock.mockClear());

describe("tabular from-a-sample flow", () => {
  it("creates the dataset, uploads the sample, and advances to the review step", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      createDataset: vi.fn().mockResolvedValue(SPEC),
      uploadSample: vi.fn().mockResolvedValue({ dataset: SPEC, profile: PROFILE }),
    });

    render(<Wizard api={api} />);
    await user.click(screen.getByRole("button", { name: /from a sample/i }));

    await user.type(screen.getByLabelText("Dataset name"), "Sampled");
    const file = new File(["a,b\n1,2"], "s.csv", { type: "text/csv" });
    await user.upload(screen.getByLabelText(/sample file/i, { selector: "input" }), file);
    await user.click(screen.getByRole("button", { name: /profile & propose schema/i }));

    await waitFor(() => expect(api.createDataset).toHaveBeenCalled());
    expect((api.createDataset as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      source: "sample",
      modality: "tabular",
    });
    await waitFor(() => expect(api.uploadSample).toHaveBeenCalledWith("ds-1", file));
    await screen.findByRole("heading", { name: /review the proposed schema/i });
  });
});

describe("text corpus flow", () => {
  it("creates a text dataset seeded by the chosen task type", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      createDataset: vi.fn().mockResolvedValue({ ...SPEC, modality: "text" }),
    });

    render(<TextWizard api={api} />);
    await user.type(screen.getByLabelText("Dataset name"), "Intents");
    await user.click(screen.getByRole("button", { name: /propose schema/i }));

    await waitFor(() => expect(api.createDataset).toHaveBeenCalled());
    const arg = (api.createDataset as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.modality).toBe("text");
    expect(arg.description).toMatch(/classification/i);
    await screen.findByRole("heading", { name: /review the proposed schema/i });
  });
});

describe("media (image) flow", () => {
  it("gates on a provider, then creates and generates, navigating to the job", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      listProviders: vi.fn().mockResolvedValue([IMAGE_PROVIDER]),
      createImageDataset: vi.fn().mockResolvedValue(SPEC),
      generate: vi.fn().mockResolvedValue(JOB),
    });

    render(<MediaWizard modality="image" api={api} />);

    // Provider list resolves → the select appears.
    const select = await screen.findByLabelText(/image provider/i);
    await user.selectOptions(select, "prov-1");
    await user.type(screen.getByLabelText("Dataset name"), "Product shots");
    await user.click(screen.getByRole("button", { name: /^generate$/i }));

    await waitFor(() => expect(api.createImageDataset).toHaveBeenCalled());
    expect((api.createImageDataset as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatchObject({
      name: "Product shots",
      directives: { model_config_id: "prov-1" },
    });
    await waitFor(() => expect(api.generate).toHaveBeenCalledWith("ds-1", { model_config_id: "prov-1" }));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/app/jobs/job-9"));
  });

  it("shows a register-provider prompt when none are configured", async () => {
    const api = baseMockApi({ listProviders: vi.fn().mockResolvedValue([]) });
    render(<MediaWizard modality="audio" api={api} />);
    expect(await screen.findByText(/no provider configured/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /register one/i })).toHaveAttribute("href", "/app/providers");
  });
});
