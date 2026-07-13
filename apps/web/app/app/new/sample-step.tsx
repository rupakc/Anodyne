"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Field, TextInput, TextArea } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import type { ApiClient, DatasetSpec, SampleProfile } from "@/lib/api";

const DEFAULT_TARGET_ROWS = 1000;

/**
 * Tabular-from-a-sample entry: create the dataset (`source: "sample"`), then
 * upload the sample file so the gateway profiles it and infers a schema
 * (`POST /datasets/{id}/sample`). The inferred spec + a short profile summary
 * are handed back to the wizard, which continues into the shared review step.
 */
export function SampleStep({
  api,
  pending,
  onCreated,
}: {
  api: ApiClient;
  pending: boolean;
  onCreated: (spec: DatasetSpec, profile: SampleProfile) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [targetRows, setTargetRows] = useState(DEFAULT_TARGET_ROWS);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = name.trim().length > 0 && file !== null && targetRows > 0;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit || !file) return;
    setBusy(true);
    setError(null);
    try {
      const created = await api.createDataset({
        name: name.trim(),
        description: description.trim() || `Profiled from ${file.name}`,
        target_rows: targetRows,
        source: "sample",
        modality: "tabular",
      });
      const result = await api.uploadSample(created.id, file);
      onCreated(result.dataset, result.profile);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to profile your sample. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const working = busy || pending;

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
    >
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Step 1 of 3
        </p>
        <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Learn the shape from a sample
        </h2>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">
          Upload a CSV or Parquet sample — Anodyne profiles it, infers a
          schema, and lets you review it before generating more like it.
        </p>
      </div>

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <Field label="Dataset name" htmlFor="sample-name">
        <TextInput
          id="sample-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Historical transactions"
          required
        />
      </Field>

      <Field label="Description" htmlFor="sample-description" hint="Optional — defaults to the file name.">
        <TextArea
          id="sample-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What does this data represent?"
        />
      </Field>

      <Field label="Sample file" htmlFor="sample-file" hint="CSV or Parquet. Only a sample is needed.">
        <input
          id="sample-file"
          type="file"
          accept=".csv,.parquet,.tsv,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          aria-required="true"
          className="rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none file:mr-3 file:rounded-md file:border-0 file:bg-secondary file:px-3 file:py-1 file:text-sm file:font-medium focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
      </Field>

      <Field label="Target row count" htmlFor="sample-target-rows" className="max-w-xs">
        <TextInput
          id="sample-target-rows"
          type="number"
          min={1}
          value={targetRows}
          onChange={(e) => setTargetRows(Number(e.target.value) || 0)}
        />
      </Field>

      <div className="flex justify-end">
        <Button type="submit" size="lg" disabled={!canSubmit || working}>
          {working ? "Profiling sample…" : "Profile & propose schema"}
        </Button>
      </div>
    </form>
  );
}
