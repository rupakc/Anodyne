// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

import { JobProgressView } from "@/app/app/jobs/[id]/job-progress-view";
import type { WebSocketLike } from "@/lib/use-job-progress";
import type { ApiClient, GenerationJob } from "@/lib/api";
import { baseMockApi } from "./mock-api";

/** A controllable fake WS: the test drives it directly via the handlers it captures. */
class FakeSocket implements WebSocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close = vi.fn();

  emitMessage(payload: { job_id: string; status: string; progress: number }) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

const INITIAL_JOB: GenerationJob = {
  id: "job-1",
  tenant_id: "tenant-1",
  dataset_id: "dataset-1",
  status: "pending",
  progress: 0,
  message: "Queued for generation.",
  workflow_id: "gen-job-1",
};

function makeMockApi(overrides: Partial<ApiClient> = {}): ApiClient {
  return baseMockApi({
    getJob: vi.fn().mockResolvedValue(INITIAL_JOB),
    ...overrides,
  });
}

describe("job progress view", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  it("renders the initial snapshot, then reflects streamed WS progress through to completion", async () => {
    const api = makeMockApi();
    let socket: FakeSocket | undefined;
    const createSocket = vi.fn(() => {
      socket = new FakeSocket();
      return socket;
    });

    render(<JobProgressView jobId="job-1" api={api} wsUrl="ws://test/jobs/job-1/stream" createSocket={createSocket} />);

    // Initial snapshot comes from GET /jobs/{id}.
    await waitFor(() => expect(api.getJob).toHaveBeenCalledWith("job-1"));
    expect(await screen.findByText("Queued")).toBeInTheDocument();
    expect(screen.getByText("Queued for generation.")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");

    expect(createSocket).toHaveBeenCalledWith("ws://test/jobs/job-1/stream");
    expect(socket).toBeDefined();

    act(() => {
      socket!.onopen?.();
    });
    expect(screen.getByTestId("connection-status")).toHaveTextContent("Live");

    // A streamed frame updates status/progress live, no polling involved.
    act(() => {
      socket!.emitMessage({ job_id: "job-1", status: "running", progress: 0.4 });
    });
    expect(screen.getByText("Generating")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");

    act(() => {
      socket!.emitMessage({ job_id: "job-1", status: "running", progress: 0.7 });
    });
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "70");

    // Completion: the socket is closed, a completion banner appears, and it
    // links to the dataset (whose id came from the initial GET /jobs/{id}).
    act(() => {
      socket!.emitMessage({ job_id: "job-1", status: "succeeded", progress: 1 });
    });
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText(/your dataset is ready/i)).toBeInTheDocument();
    expect(socket!.close).toHaveBeenCalled();

    const link = screen.getByRole("link", { name: /view versions/i });
    expect(link).toHaveAttribute("href", "/app/datasets/dataset-1");
  });

  it("falls back to polling GET /jobs/{id} when the socket closes, and stops polling once terminal", async () => {
    vi.useFakeTimers();
    const runningJob: GenerationJob = { ...INITIAL_JOB, status: "running", progress: 0.5 };
    const succeededJob: GenerationJob = { ...INITIAL_JOB, status: "succeeded", progress: 1 };

    const getJob = vi
      .fn()
      .mockResolvedValueOnce(INITIAL_JOB) // initial snapshot
      .mockResolvedValueOnce(runningJob) // first poll
      .mockResolvedValueOnce(succeededJob); // second poll: terminal, stops the interval

    const api = makeMockApi({ getJob });
    let socket: FakeSocket | undefined;
    const createSocket = vi.fn(() => {
      socket = new FakeSocket();
      return socket;
    });

    render(
      <JobProgressView
        jobId="job-1"
        api={api}
        wsUrl="ws://test/jobs/job-1/stream"
        createSocket={createSocket}
      />,
    );

    await vi.waitFor(() => expect(getJob).toHaveBeenCalledTimes(1));

    // Socket drops before ever streaming anything -> fall back to polling.
    // `startPolling` fires an immediate poll (call #2) before the interval's
    // first tick, so that's already in flight once `onclose` returns.
    await act(async () => {
      socket!.onclose?.();
    });
    expect(screen.getByTestId("connection-status")).toHaveTextContent(/polling/i);
    await vi.waitFor(() => expect(getJob).toHaveBeenCalledTimes(2));
    expect(screen.getByText("Generating")).toBeInTheDocument();

    // The interval's first tick (call #3) reports the terminal status.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(getJob).toHaveBeenCalledTimes(3);
    expect(screen.getByText("Complete")).toBeInTheDocument();

    // Polling stopped: no further calls after the terminal status.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(getJob).toHaveBeenCalledTimes(3);

    vi.useRealTimers();
  });
});
