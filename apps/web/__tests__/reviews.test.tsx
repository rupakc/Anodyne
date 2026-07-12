// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

afterEach(cleanup);

const refreshMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh: refreshMock, push: vi.fn() }),
}));

import { ReviewQueue } from "@/app/app/reviews/review-queue";
import { ReviewDetail } from "@/app/app/reviews/[id]/review-detail";
import { ApiError, type ReviewItem } from "@/lib/api";
import { baseMockApi } from "./mock-api";

const PENDING: ReviewItem[] = [
  {
    id: "review-1",
    status: "pending",
    kind: "schema_approval",
    target_type: "dataset",
    target_id: "dataset-1",
    title: "Approve proposed schema",
    summary: "6 fields proposed for Support tickets.",
    payload: { fields: ["priority", "created_at"] },
  },
];

describe("review queue", () => {
  it("renders pending reviews from a mocked api client", async () => {
    const api = baseMockApi({ listReviews: vi.fn().mockResolvedValue(PENDING) });
    render(<ReviewQueue api={api} />);

    expect(await screen.findByText("Approve proposed schema")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /approve proposed schema/i });
    expect(link).toHaveAttribute("href", "/app/reviews/review-1");
  });

  it("degrades to a friendly empty state when the HITL routes 404", async () => {
    const api = baseMockApi({
      listReviews: vi.fn().mockRejectedValue(new ApiError(404, "not found")),
    });
    render(<ReviewQueue api={api} />);

    expect(await screen.findByText(/aren't enabled for this tenant/i)).toBeInTheDocument();
  });

  it("shows an empty queue message when nothing is pending", async () => {
    const api = baseMockApi({ listReviews: vi.fn().mockResolvedValue([]) });
    render(<ReviewQueue api={api} />);

    expect(await screen.findByText(/the queue is clear/i)).toBeInTheDocument();
  });
});

describe("review detail", () => {
  beforeEach(() => refreshMock.mockClear());

  it("renders the review payload and submits an approve decision", async () => {
    const user = userEvent.setup();
    const submit = vi.fn().mockResolvedValue({ ...PENDING[0], status: "approved" });
    const api = baseMockApi({
      getReview: vi.fn().mockResolvedValue(PENDING[0]),
      submitReviewDecision: submit,
    });

    render(<ReviewDetail reviewId="review-1" api={api} />);

    expect(await screen.findByRole("heading", { name: /approve proposed schema/i })).toBeInTheDocument();
    await user.type(screen.getByLabelText(/comment/i), "looks good");
    await user.click(screen.getByRole("button", { name: /^approve$/i }));

    await waitFor(() =>
      expect(submit).toHaveBeenCalledWith("review-1", { decision: "approve", comment: "looks good" }),
    );
    expect(await screen.findByText(/decision recorded/i)).toBeInTheDocument();
  });

  it("surfaces an inline error if submitting the decision fails", async () => {
    const user = userEvent.setup();
    const api = baseMockApi({
      getReview: vi.fn().mockResolvedValue(PENDING[0]),
      submitReviewDecision: vi.fn().mockRejectedValue(new Error("decision route down")),
    });

    render(<ReviewDetail reviewId="review-1" api={api} />);
    await screen.findByRole("heading", { name: /approve proposed schema/i });
    await user.click(screen.getByRole("button", { name: /reject/i }));

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/decision route down/i)).toBeInTheDocument();
  });
});
