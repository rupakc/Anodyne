"use client";

import { useMemo, useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { createApiClient, type ApiClient } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface FeedbackWidgetProps {
  targetType: string;
  targetId: string;
  accessToken?: string;
  api?: ApiClient;
  /** Optional label rendered before the controls. */
  label?: string;
}

/**
 * Thumbs-up/down + optional comment feedback, posted to `POST /feedback`.
 * Best-effort: if the HITL feedback route isn't wired yet the widget shows
 * an inline note rather than blocking the surrounding view.
 */
export function FeedbackWidget({
  targetType,
  targetId,
  accessToken,
  api: injectedApi,
  label = "Was this useful?",
}: FeedbackWidgetProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [thumbs, setThumbs] = useState<"up" | "down" | null>(null);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "sent" | "error">("idle");

  async function send(next: "up" | "down") {
    setThumbs(next);
    setStatus("saving");
    try {
      await api.submitFeedback({
        target_type: targetType,
        target_id: targetId,
        thumbs: next,
        comment: comment.trim() || undefined,
      });
      setStatus("sent");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-muted/40 px-4 py-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <div className="flex items-center gap-1.5" role="group" aria-label="Feedback">
        <button
          type="button"
          aria-label="Thumbs up"
          aria-pressed={thumbs === "up"}
          onClick={() => send("up")}
          className={cn(
            "rounded-lg border p-1.5 transition-colors",
            thumbs === "up"
              ? "border-sage bg-sage/20 text-foreground"
              : "border-border bg-background text-muted-foreground hover:bg-muted",
          )}
        >
          <ThumbsUp className="size-4" />
        </button>
        <button
          type="button"
          aria-label="Thumbs down"
          aria-pressed={thumbs === "down"}
          onClick={() => send("down")}
          className={cn(
            "rounded-lg border p-1.5 transition-colors",
            thumbs === "down"
              ? "border-destructive/50 bg-destructive/10 text-destructive"
              : "border-border bg-background text-muted-foreground hover:bg-muted",
          )}
        >
          <ThumbsDown className="size-4" />
        </button>
      </div>
      <input
        aria-label="Feedback comment"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Optional comment"
        className="min-w-40 flex-1 rounded-lg border border-input bg-background px-3 py-1.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50"
      />
      <span aria-live="polite" className="text-xs text-muted-foreground">
        {status === "saving"
          ? "Saving…"
          : status === "sent"
            ? "Thanks!"
            : status === "error"
              ? "Feedback endpoint unavailable"
              : ""}
      </span>
    </div>
  );
}
