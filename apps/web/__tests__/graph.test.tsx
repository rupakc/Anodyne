// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

import { GraphWizard } from "@/app/app/new/graph-wizard";
import { GraphExplorer } from "@/components/graph-explorer";
import { OntologyView } from "@/components/ontology-view";
import { ExportPanel } from "@/app/app/datasets/[id]/export-panel";
import { baseMockApi } from "./mock-api";
import type { DatasetSpec, GenerationJob, GraphArtifact, ProviderConfig } from "@/lib/api";

const ONTOLOGY = {
  node_types: [
    { name: "Gene", properties: [{ name: "symbol", datatype: "string" }] },
    { name: "Disease", properties: [{ name: "name", datatype: "string" }], subclass_of: "Entity" },
  ],
  edge_types: [{ name: "associated_with", source_type: "Gene", target_type: "Disease" }],
};

const GRAPH_SPEC: DatasetSpec = {
  id: "g-1",
  tenant_id: "t-1",
  name: "Biomed KG",
  description: "genes and diseases",
  modality: "graph",
  source: "description",
  target_rows: 500,
  directives: { ontology: ONTOLOGY, topology: { model: "ontology_constrained" } },
  status: "draft",
  created_at: "2026-07-13T00:00:00Z",
  fields: [],
};

const JOB: GenerationJob = {
  id: "job-7",
  tenant_id: "t-1",
  dataset_id: "g-1",
  status: "pending",
  progress: 0,
  message: "",
  workflow_id: null,
};

const MODEL: ProviderConfig = {
  id: "m-1",
  tenant_id: "t-1",
  name: "GPT",
  provider: "openai",
  model: "gpt-4o",
  params: {},
  api_base: null,
  enabled: true,
};

const FIXTURE: GraphArtifact = {
  ontology: ONTOLOGY,
  nodes: [
    { id: "n1", type: "Gene", properties: { symbol: "BRCA1" } },
    { id: "n2", type: "Gene", properties: { symbol: "TP53" } },
    { id: "n3", type: "Disease", properties: { name: "Cancer" } },
  ],
  edges: [
    { id: "e1", type: "associated_with", source: "n1", target: "n3" },
    { id: "e2", type: "associated_with", source: "n2", target: "n3" },
  ],
  metrics: { node_count: 3, edge_count: 2, nodes_by_type: { Gene: 2, Disease: 1 } },
};

beforeEach(() => pushMock.mockClear());

describe("graph wizard", () => {
  it("creates a graph dataset and renders the proposed ontology to review", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      createGraphDataset: vi.fn().mockResolvedValue(GRAPH_SPEC),
    });

    render(<GraphWizard api={api} />);
    await user.type(screen.getByLabelText("Dataset name"), "Biomed KG");
    await user.click(screen.getByRole("button", { name: /propose ontology/i }));

    await waitFor(() => expect(api.createGraphDataset).toHaveBeenCalled());
    const arg = (api.createGraphDataset as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg).toMatchObject({ name: "Biomed KG", source: "description" });

    await screen.findByRole("heading", { name: /review the proposed ontology/i });
    // Ontology renders node + edge types.
    const nodeTypes = screen.getByLabelText("Node types");
    expect(within(nodeTypes).getByText("Gene")).toBeInTheDocument();
    const edgeTypes = screen.getByLabelText("Edge types");
    expect(within(edgeTypes).getByText("associated_with")).toBeInTheDocument();
  });

  it("proceeds to confirm and starts generation, navigating to the job", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      createGraphDataset: vi.fn().mockResolvedValue(GRAPH_SPEC),
      listProviders: vi.fn().mockResolvedValue([MODEL]),
      generate: vi.fn().mockResolvedValue(JOB),
    });

    render(<GraphWizard api={api} />);
    await user.type(screen.getByLabelText("Dataset name"), "Biomed KG");
    await user.click(screen.getByRole("button", { name: /propose ontology/i }));
    await screen.findByRole("heading", { name: /review the proposed ontology/i });
    await user.click(screen.getByRole("button", { name: /save & continue/i }));

    await screen.findByRole("heading", { name: /confirm & generate/i });
    await user.click(screen.getByRole("button", { name: /generate graph/i }));

    await waitFor(() => expect(api.generate).toHaveBeenCalledWith("g-1", expect.objectContaining({ require_review: false })));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/app/jobs/job-7"));
  });
});

describe("ontology view", () => {
  it("renders node types, properties and edge source→target", () => {
    render(<OntologyView ontology={ONTOLOGY} />);
    const nodeTypes = screen.getByLabelText("Node types");
    expect(within(nodeTypes).getByText("Gene")).toBeInTheDocument();
    expect(within(nodeTypes).getByText("Disease")).toBeInTheDocument();
    expect(within(nodeTypes).getByText(/symbol/)).toBeInTheDocument();
    // Edge type row shows the relation and both endpoints.
    const edges = screen.getByLabelText("Edge types");
    expect(within(edges).getByText("associated_with")).toBeInTheDocument();
    // Subclass hierarchy surfaces when present.
    expect(screen.getByLabelText("Subclass hierarchy")).toBeInTheDocument();
  });
});

describe("graph explorer", () => {
  it("draws the nodes and edges of a small graph and reports full-graph counts", () => {
    render(<GraphExplorer graph={FIXTURE} />);

    // Canvas is present with an accessible summary label.
    expect(screen.getByRole("img", { name: /force-directed graph/i })).toBeInTheDocument();

    // Node + edge mirrors are rendered (accessible + testable, no canvas needed).
    const nodes = screen.getByLabelText("Graph nodes");
    expect(within(nodes).getAllByRole("listitem")).toHaveLength(3);
    const edges = screen.getByLabelText("Graph edges");
    expect(within(edges).getAllByRole("listitem")).toHaveLength(2);

    // Legend surfaces per-type counts from metrics.
    const legend = screen.getByLabelText("Node type legend");
    expect(within(legend).getByText("Gene")).toBeInTheDocument();
  });
});

describe("export panel — graph formats", () => {
  it("offers graph formats and calls exportGraphVersion", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      exportGraphVersion: vi.fn().mockResolvedValue("graph.ttl"),
    });

    render(<ExportPanel api={api} datasetId="g-1" versionId="v-1" rowCount={1000} modality="graph" />);

    const select = screen.getByLabelText("Format");
    await user.selectOptions(select, "turtle");
    await user.click(screen.getByRole("button", { name: /^export$/i }));

    await waitFor(() =>
      expect(api.exportGraphVersion).toHaveBeenCalledWith("g-1", "v-1", "turtle"),
    );
    expect(await screen.findByText(/graph\.ttl/)).toBeInTheDocument();
    // Non-graph export path is untouched.
    expect(api.exportVersion).not.toHaveBeenCalled();
  });
});
