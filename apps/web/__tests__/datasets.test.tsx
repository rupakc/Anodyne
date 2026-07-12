// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

import { DatasetList } from "@/app/app/datasets/dataset-list";
import { DatasetVersions } from "@/app/app/datasets/[id]/dataset-versions";
import type { ApiClient, DatasetSpec, DatasetVersion } from "@/lib/api";
import { baseMockApi } from "./mock-api";

function makeMockApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return baseMockApi({
    listDatasets: vi.fn().mockResolvedValue([]),
    listVersions: vi.fn().mockResolvedValue([]),
    listTemplates: vi.fn().mockResolvedValue([]),
    ...overrides,
  });
}

const DATASETS: DatasetSpec[] = [
  {
    id: "dataset-1",
    tenant_id: "tenant-1",
    name: "Support tickets",
    description: "Customer support tickets with a priority",
    modality: "tabular",
    source: "description",
    fields: [],
    target_rows: 500,
    directives: {},
    status: "ready",
    created_at: "2026-07-01T00:00:00Z",
  },
  {
    id: "dataset-2",
    tenant_id: "tenant-1",
    name: "Fraud transactions",
    description: "Synthetic card transactions with a fraud flag",
    modality: "tabular",
    source: "description",
    fields: [],
    target_rows: 10000,
    directives: {},
    status: "draft",
    created_at: "2026-07-02T00:00:00Z",
  },
];

const VERSIONS: DatasetVersion[] = [
  {
    id: "version-1",
    tenant_id: "tenant-1",
    dataset_id: "dataset-1",
    artifact_uri: "datasets/dataset-1/version-1.parquet",
    format: "parquet",
    row_count: 500,
    checksum: "abcdef1234567890",
    created_at: "2026-07-01T01:00:00Z",
  },
  {
    id: "version-2",
    tenant_id: "tenant-1",
    dataset_id: "dataset-1",
    artifact_uri: "datasets/dataset-1/version-2.parquet",
    format: "parquet",
    row_count: 750,
    checksum: "0987654321fedcba",
    created_at: "2026-07-05T01:00:00Z",
  },
];

describe("dataset list", () => {
  it("renders every dataset from a mocked api client, linking to its detail page", async () => {
    const api = makeMockApi({ listDatasets: vi.fn().mockResolvedValue(DATASETS) });

    render(<DatasetList api={api} />);

    expect(await screen.findByText("Support tickets")).toBeInTheDocument();
    expect(screen.getByText("Fraud transactions")).toBeInTheDocument();
    expect(screen.getByText("500 rows")).toBeInTheDocument();
    expect(screen.getByText("10,000 rows")).toBeInTheDocument();

    const link = screen.getByRole("link", { name: /support tickets/i });
    expect(link).toHaveAttribute("href", "/app/datasets/dataset-1");
  });

  it("shows an empty state when there are no datasets yet", async () => {
    const api = makeMockApi({ listDatasets: vi.fn().mockResolvedValue([]) });

    render(<DatasetList api={api} />);

    expect(await screen.findByText(/no datasets yet/i)).toBeInTheDocument();
  });

  it("surfaces an inline error if the list fails to load", async () => {
    const api = makeMockApi({ listDatasets: vi.fn().mockRejectedValue(new Error("gateway unavailable")) });

    render(<DatasetList api={api} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("gateway unavailable");
  });
});

describe("dataset versions + download", () => {
  it("renders the dataset and its versions from a mocked api client", async () => {
    const api = makeMockApi({
      getDataset: vi.fn().mockResolvedValue(DATASETS[0]),
      listVersions: vi.fn().mockResolvedValue(VERSIONS),
    });

    render(<DatasetVersions datasetId="dataset-1" api={api} />);

    expect(await screen.findByRole("heading", { name: /support tickets/i })).toBeInTheDocument();
    const list = await screen.findByRole("list", { name: /dataset versions/i });
    expect(within(list).getByText(/PARQUET · 500 rows/)).toBeInTheDocument();
    expect(within(list).getByText(/PARQUET · 750 rows/)).toBeInTheDocument();
  });

  it("shows an empty state when generation hasn't produced a version yet", async () => {
    const api = makeMockApi({
      getDataset: vi.fn().mockResolvedValue(DATASETS[0]),
      listVersions: vi.fn().mockResolvedValue([]),
    });

    render(<DatasetVersions datasetId="dataset-1" api={api} />);

    expect(await screen.findByText(/no versions yet/i)).toBeInTheDocument();
  });

  it("downloads a version by resolving its presigned URL and opening it", async () => {
    const user = userEvent.setup();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const api = makeMockApi({
      getDataset: vi.fn().mockResolvedValue(DATASETS[0]),
      listVersions: vi.fn().mockResolvedValue(VERSIONS),
      downloadUrl: vi.fn().mockResolvedValue("https://minio.local/presigned/version-1?sig=abc"),
    });

    render(<DatasetVersions datasetId="dataset-1" api={api} />);

    const list = await screen.findByRole("list", { name: /dataset versions/i });
    const firstRow = within(list).getByText(/PARQUET · 500 rows/).closest("li")!;
    await user.click(within(firstRow).getByRole("button", { name: /download/i }));

    await waitFor(() => expect(api.downloadUrl).toHaveBeenCalledWith("dataset-1", "version-1"));
    await waitFor(() =>
      expect(openSpy).toHaveBeenCalledWith(
        "https://minio.local/presigned/version-1?sig=abc",
        "_blank",
        "noopener,noreferrer",
      ),
    );

    openSpy.mockRestore();
  });

  it("surfaces an inline error if getting the download link fails", async () => {
    const user = userEvent.setup();
    const api = makeMockApi({
      getDataset: vi.fn().mockResolvedValue(DATASETS[0]),
      listVersions: vi.fn().mockResolvedValue(VERSIONS),
      downloadUrl: vi.fn().mockRejectedValue(new Error("object not found")),
    });

    render(<DatasetVersions datasetId="dataset-1" api={api} />);

    const list = await screen.findByRole("list", { name: /dataset versions/i });
    const firstRow = within(list).getByText(/PARQUET · 500 rows/).closest("li")!;
    await user.click(within(firstRow).getByRole("button", { name: /download/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("object not found");
  });
});
