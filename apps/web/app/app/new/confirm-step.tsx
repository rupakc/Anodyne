"use client";

import { Button } from "@/components/ui/button";
import type { DatasetSpec } from "@/lib/api";

export function ConfirmStep({
  spec,
  pending,
  requireReview,
  onRequireReviewChange,
  onBack,
  onGenerate,
}: {
  spec: DatasetSpec;
  pending: boolean;
  requireReview: boolean;
  onRequireReviewChange: (value: boolean) => void;
  onBack: () => void;
  onGenerate: () => void;
}) {
  return (
    <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Step 3 of 3
        </p>
        <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Confirm & generate
        </h2>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">
          Double-check the shape below, then kick off generation. You&apos;ll be able to track
          progress on the next screen.
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-4 rounded-xl border border-border bg-muted/40 p-5 text-sm sm:grid-cols-3">
        <div>
          <dt className="text-xs tracking-wide text-muted-foreground uppercase">Name</dt>
          <dd className="mt-1 font-medium">{spec.name}</dd>
        </div>
        <div>
          <dt className="text-xs tracking-wide text-muted-foreground uppercase">Fields</dt>
          <dd className="mt-1 font-medium">{spec.fields.length}</dd>
        </div>
        <div>
          <dt className="text-xs tracking-wide text-muted-foreground uppercase">Target rows</dt>
          <dd className="mt-1 font-medium">{spec.target_rows.toLocaleString()}</dd>
        </div>
      </dl>

      <ul className="flex flex-wrap gap-2" aria-label="Proposed fields">
        {spec.fields.map((field, index) => (
          <li
            key={`${field.name}-${index}`}
            className="rounded-full border border-border bg-background px-3 py-1 font-[family-name:var(--font-data)] text-xs text-muted-foreground"
          >
            {field.name} <span className="text-terracotta">· {field.semantic_type}</span>
          </li>
        ))}
      </ul>

      <label className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 p-4 text-sm">
        <input
          type="checkbox"
          className="mt-0.5 size-4 accent-terracotta"
          checked={requireReview}
          disabled={pending}
          onChange={(event) => onRequireReviewChange(event.target.checked)}
        />
        <span>
          <span className="font-medium">Require human review before generating</span>
          <span className="mt-1 block text-muted-foreground text-pretty">
            Park this job for a reviewer to approve in the reviews queue before any data is
            generated. Off by default.
          </span>
        </span>
      </label>

      <div className="flex justify-between">
        <Button type="button" variant="outline" onClick={onBack} disabled={pending}>
          Back
        </Button>
        <Button type="button" size="lg" onClick={onGenerate} disabled={pending}>
          {pending ? "Starting generation…" : "Generate dataset"}
        </Button>
      </div>
    </section>
  );
}
