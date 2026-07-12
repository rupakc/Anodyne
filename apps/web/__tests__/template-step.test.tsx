// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

import { TemplateStep } from "@/app/app/new/template-step";
import type { ApiClient, DatasetSpec, DatasetTemplate } from "@/lib/api";

const TEMPLATES: DatasetTemplate[] = [
  {
    key: "customers",
    name: "Customers",
    description: "A customer roster.",
    category: "crm",
    modality: "tabular",
    fields: [],
    default_target_rows: 1000,
    default_directives: {},
  },
  {
    key: "users_churn",
    name: "Users + churn label",
    description: "User accounts with a churn label.",
    category: "ml-labels",
    modality: "tabular",
    fields: [],
    default_target_rows: 3000,
    default_directives: {},
  },
];

const CREATED_SPEC: DatasetSpec = {
  id: "dataset-9",
  tenant_id: "tenant-1",
  name: "Customers",
  description: "A customer roster.",
  modality: "tabular",
  source: "template",
  target_rows: 1000,
  directives: {},
  status: "draft",
  created_at: "2026-07-12T00:00:00Z",
  fields: [
    { name: "full_name", semantic_type: "name", nullable: false, constraints: {}, distribution: null },
  ],
};

function makeMockApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    createDataset: vi.fn(),
    listDatasets: vi.fn(),
    getDataset: vi.fn(),
    updateDataset: vi.fn(),
    generate: vi.fn(),
    getJob: vi.fn(),
    listVersions: vi.fn(),
    downloadUrl: vi.fn(),
    listTemplates: vi.fn().mockResolvedValue(TEMPLATES),
    createFromTemplate: vi.fn().mockResolvedValue(CREATED_SPEC),
    ...overrides,
  };
}

describe("TemplateStep", () => {
  it("lists templates and creates a dataset from the selected one", async () => {
    const user = userEvent.setup();
    const api = makeMockApi();
    const onCreated = vi.fn();

    render(<TemplateStep api={api} pending={false} onCreated={onCreated} />);

    expect(await screen.findByText("Customers")).toBeInTheDocument();
    expect(screen.getByText("Users + churn label")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /customers/i }));

    await waitFor(() =>
      expect(api.createFromTemplate).toHaveBeenCalledWith({ template_key: "customers" }),
    );
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(CREATED_SPEC));
  });

  it("surfaces an inline error when the catalog fails to load", async () => {
    const api = makeMockApi({
      listTemplates: vi.fn().mockRejectedValue(new Error("gateway unavailable")),
    });

    render(<TemplateStep api={api} pending={false} onCreated={vi.fn()} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("gateway unavailable");
  });

  it("surfaces an inline error when creating from a template fails", async () => {
    const user = userEvent.setup();
    const api = makeMockApi({
      createFromTemplate: vi.fn().mockRejectedValue(new Error("could not create dataset")),
    });
    const onCreated = vi.fn();

    render(<TemplateStep api={api} pending={false} onCreated={onCreated} />);
    await user.click(await screen.findByRole("button", { name: /customers/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("could not create dataset");
    expect(onCreated).not.toHaveBeenCalled();
  });
});
