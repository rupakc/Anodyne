"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import type { CreateDatasetInput } from "@/lib/api";

const DEFAULT_TARGET_ROWS = 1000;

export function DescribeStep({
  pending,
  onSubmit,
}: {
  pending: boolean;
  onSubmit: (input: CreateDatasetInput) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [targetRows, setTargetRows] = useState(DEFAULT_TARGET_ROWS);

  const canSubmit = name.trim().length > 0 && description.trim().length > 0 && targetRows > 0;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({ name: name.trim(), description: description.trim(), target_rows: targetRows });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
    >
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Step 1 of 3
        </p>
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Describe the dataset you want
        </h1>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">
          Plain English is fine — the gateway proposes a schema from your
          description, and you review it next.
        </p>
      </div>

      <label className="flex flex-col gap-2 text-sm font-medium" htmlFor="dataset-name">
        Dataset name
        <input
          id="dataset-name"
          className="rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Customer support tickets"
          required
        />
      </label>

      <label className="flex flex-col gap-2 text-sm font-medium" htmlFor="dataset-description">
        Description
        <textarea
          id="dataset-description"
          className="min-h-32 rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the rows, fields, and any constraints you care about..."
          required
        />
      </label>

      <label className="flex max-w-xs flex-col gap-2 text-sm font-medium" htmlFor="dataset-target-rows">
        Target row count
        <input
          id="dataset-target-rows"
          type="number"
          min={1}
          className="rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          value={targetRows}
          onChange={(e) => setTargetRows(Number(e.target.value) || 0)}
        />
      </label>

      <div className="flex justify-end">
        <Button type="submit" size="lg" disabled={!canSubmit || pending}>
          {pending ? "Proposing schema…" : "Propose schema"}
        </Button>
      </div>
    </form>
  );
}
