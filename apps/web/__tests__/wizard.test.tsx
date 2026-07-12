// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// @testing-library/react's auto-cleanup only self-registers when it detects
// a global `afterEach` (i.e. `test.globals: true` in the Vitest config);
// this project explicitly imports test globals instead, so register
// cleanup ourselves to avoid leaking DOM between the tests below.
afterEach(cleanup);

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

import { Wizard } from "@/app/app/new/wizard";
import type { ApiClient, DatasetSpec, GenerationJob } from "@/lib/api";

const PROPOSED_SPEC: DatasetSpec = {
  id: "dataset-1",
  tenant_id: "tenant-1",
  name: "Support tickets",
  description: "Customer support tickets with a priority and a resolved flag",
  modality: "tabular",
  source: "description",
  target_rows: 500,
  directives: {},
  status: "draft",
  created_at: "2026-07-12T00:00:00Z",
  fields: [
    { name: "priority", semantic_type: "categorical", nullable: false, constraints: {}, distribution: null },
    { name: "created_at", semantic_type: "datetime", nullable: false, constraints: {}, distribution: null },
  ],
};

const GENERATED_JOB: GenerationJob = {
  id: "job-1",
  tenant_id: "tenant-1",
  dataset_id: "dataset-1",
  status: "pending",
  progress: 0,
  message: "",
  workflow_id: "gen-job-1",
};

function makeMockApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    createDataset: vi.fn().mockResolvedValue(PROPOSED_SPEC),
    listDatasets: vi.fn(),
    getDataset: vi.fn(),
    updateDataset: vi.fn().mockImplementation(async (id: string, patch) => ({
      ...PROPOSED_SPEC,
      ...patch,
    })),
    generate: vi.fn().mockResolvedValue(GENERATED_JOB),
    getJob: vi.fn(),
    listVersions: vi.fn(),
    downloadUrl: vi.fn(),
    ...overrides,
  };
}

async function fillDescribeStep(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Dataset name"), "Support tickets");
  await user.type(
    screen.getByLabelText("Description"),
    "Customer support tickets with a priority and a resolved flag",
  );
  const rowsInput = screen.getByLabelText("Target row count");
  await user.clear(rowsInput);
  await user.type(rowsInput, "500");
  await user.click(screen.getByRole("button", { name: /propose schema/i }));
}

describe("create-from-description wizard", () => {
  beforeEach(() => {
    pushMock.mockClear();
  });

  it("walks describe -> review (showing the proposed fields) -> confirm -> generate", async () => {
    const user = userEvent.setup();
    const api = makeMockApi();

    render(<Wizard api={api} />);

    // Step 1: describe. Nothing proposed yet.
    expect(screen.getByRole("heading", { name: /describe the dataset/i })).toBeInTheDocument();
    await fillDescribeStep(user);

    expect(api.createDataset).toHaveBeenCalledWith({
      name: "Support tickets",
      description: "Customer support tickets with a priority and a resolved flag",
      target_rows: 500,
    });

    // Step 2: review. The mocked proposal's fields render as editable rows.
    await screen.findByRole("heading", { name: /review the proposed schema/i });
    expect(screen.getByLabelText("Field 1 name")).toHaveValue("priority");
    expect(screen.getByLabelText("Field 1 semantic type")).toHaveValue("categorical");
    expect(screen.getByLabelText("Field 2 name")).toHaveValue("created_at");

    // Edit a field's semantic type.
    await user.selectOptions(screen.getByLabelText("Field 1 semantic type"), "text");
    expect(screen.getByLabelText("Field 1 semantic type")).toHaveValue("text");

    await user.click(screen.getByRole("button", { name: /save & continue/i }));

    // The edited spec (with the changed field) is what gets persisted.
    await waitFor(() => expect(api.updateDataset).toHaveBeenCalledTimes(1));
    const [calledId, calledPatch] = (api.updateDataset as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(calledId).toBe("dataset-1");
    expect(calledPatch.fields[0]).toMatchObject({ name: "priority", semantic_type: "text" });
    expect(calledPatch.fields[1]).toMatchObject({ name: "created_at", semantic_type: "datetime" });

    // Step 3: confirm + generate. The edited type shows up in the summary.
    await screen.findByRole("heading", { name: /confirm & generate/i });
    const summary = screen.getByRole("list", { name: /proposed fields/i });
    expect(within(summary).getByText(/priority/)).toBeInTheDocument();
    expect(within(summary).getByText(/· text/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /generate dataset/i }));

    await waitFor(() => expect(api.generate).toHaveBeenCalledWith("dataset-1"));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/app/jobs/job-1"));
  });

  it("supports going back from review to describe without losing the proposal", async () => {
    const user = userEvent.setup();
    const api = makeMockApi();

    render(<Wizard api={api} />);
    await fillDescribeStep(user);
    await screen.findByRole("heading", { name: /review the proposed schema/i });

    await user.click(screen.getByRole("button", { name: /^back$/i }));
    expect(screen.getByRole("heading", { name: /describe the dataset/i })).toBeInTheDocument();
  });

  it("surfaces an inline error and stays on the describe step when schema proposal fails", async () => {
    const user = userEvent.setup();
    const api = makeMockApi({
      createDataset: vi.fn().mockRejectedValue(new Error("gateway unavailable")),
    });

    render(<Wizard api={api} />);
    await fillDescribeStep(user);

    expect(await screen.findByRole("alert")).toHaveTextContent("gateway unavailable");
    expect(screen.getByRole("heading", { name: /describe the dataset/i })).toBeInTheDocument();
    expect(api.updateDataset).not.toHaveBeenCalled();
    expect(api.generate).not.toHaveBeenCalled();
  });
});
