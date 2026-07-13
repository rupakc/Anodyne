"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import {
  ApiError,
  createApiClient,
  type ApiClient,
  type ReviewDecision,
  type ReviewItem,
} from "@/lib/api";
import { EmptyState, ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";
import { Field, TextArea } from "@/components/ui/form";
import { Button } from "@/components/ui/button";

export interface ReviewDetailProps {
  reviewId: string;
  accessToken?: string;
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error" | "unavailable";

const DECISIONS: { decision: ReviewDecision; label: string; variant: "default" | "outline" | "destructive" }[] = [
  { decision: "approve", label: "Approve", variant: "default" },
  { decision: "changes_requested", label: "Request changes", variant: "outline" },
  { decision: "reject", label: "Reject", variant: "destructive" },
];

/**
 * Single review with a full payload dump and the approve / request-changes /
 * reject controls (`POST /reviews/{id}/decision`). Tolerates a missing HITL
 * backend (404) gracefully.
 */
export function ReviewDetail({ reviewId, accessToken, api: injectedApi }: ReviewDetailProps) {
  const router = useRouter();
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [item, setItem] = useState<ReviewItem | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState<ReviewDecision | null>(null);
  const [decided, setDecided] = useState<ReviewDecision | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.getReview(reviewId).then(
      (r) => {
        if (cancelled) return;
        setItem(r);
        setState("ready");
      },
      (err) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setState("unavailable");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load this review.");
        setState("error");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, reviewId]);

  async function decide(decision: ReviewDecision) {
    setSubmitting(decision);
    setSubmitError(null);
    try {
      const updated = await api.submitReviewDecision(reviewId, {
        decision,
        comment: comment.trim() || undefined,
      });
      setItem(updated);
      setDecided(decision);
      router.refresh();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to submit your decision.");
    } finally {
      setSubmitting(null);
    }
  }

  const back = (
    <Link
      href="/app/reviews"
      className="inline-flex items-center gap-1.5 text-sm text-terracotta underline-offset-4 hover:underline"
    >
      <ArrowLeft className="size-4" />
      Back to queue
    </Link>
  );

  if (state === "loading") return <Loading label="Loading review…" />;
  if (state === "unavailable")
    return (
      <div className="flex flex-col gap-4">
        {back}
        <EmptyState>This review is no longer available.</EmptyState>
      </div>
    );
  if (state === "error")
    return (
      <div className="flex flex-col gap-4">
        {back}
        <ErrorAlert>{error}</ErrorAlert>
      </div>
    );

  const payload = item?.payload ?? {};

  return (
    <div className="flex flex-col gap-6">
      {back}
      <SectionHeading
        eyebrow={item?.kind ?? "Review"}
        title={item?.title ?? `Review ${reviewId.slice(0, 8)}`}
        description={item?.summary}
      />

      {item?.target_type ? (
        <p className="font-[family-name:var(--font-data)] text-xs text-sage uppercase">
          {item.target_type}
          {item.target_id ? ` · ${item.target_id}` : ""}
        </p>
      ) : null}

      {Object.keys(payload).length > 0 ? (
        <div className="overflow-x-auto rounded-2xl border border-border bg-card p-5 shadow-sm">
          <pre className="font-[family-name:var(--font-data)] text-xs whitespace-pre">
            {JSON.stringify(payload, null, 2)}
          </pre>
        </div>
      ) : null}

      {decided ? (
        <div className="rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm">
          <p className="font-medium text-foreground">Decision recorded: {decided.replace(/_/g, " ")}.</p>
          <p className="mt-1 text-muted-foreground">
            <Link href="/app/reviews" className="text-terracotta underline underline-offset-4">
              Return to the queue
            </Link>{" "}
            for the next item.
          </p>
        </div>
      ) : (
        <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <h2 className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Your decision
          </h2>
          <Field label="Comment (optional)" htmlFor="review-comment">
            <TextArea
              id="review-comment"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Explain your decision, especially when requesting changes or rejecting…"
            />
          </Field>
          {submitError ? <ErrorAlert>{submitError}</ErrorAlert> : null}
          <div className="flex flex-wrap gap-3">
            {DECISIONS.map((d) => (
              <Button
                key={d.decision}
                type="button"
                size="lg"
                variant={d.variant}
                disabled={submitting !== null}
                onClick={() => decide(d.decision)}
              >
                {submitting === d.decision ? "Submitting…" : d.label}
              </Button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
