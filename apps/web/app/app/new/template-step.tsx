"use client";

import { useEffect, useState } from "react";
import type { ApiClient, DatasetSpec, DatasetTemplate } from "@/lib/api";

/**
 * Alternative entry point for wizard step 1 (see `wizard.tsx`'s `mode` toggle):
 * pick a ready-made template instead of describing a dataset from scratch.
 * Selecting one calls `POST /datasets/from-template` and hands the created
 * spec back to the wizard, which joins the same `review` step the
 * describe-from-scratch flow lands on.
 */
export function TemplateStep({
  api,
  pending,
  onCreated,
}: {
  api: ApiClient;
  pending: boolean;
  onCreated: (spec: DatasetSpec) => void;
}) {
  const [templates, setTemplates] = useState<DatasetTemplate[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creatingKey, setCreatingKey] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.listTemplates().then(
      (list) => {
        if (!cancelled) setTemplates(list);
      },
      (err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load templates.");
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api]);

  async function handleSelect(template: DatasetTemplate) {
    setCreatingKey(template.key);
    setError(null);
    try {
      const spec = await api.createFromTemplate({ template_key: template.key });
      onCreated(spec);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create dataset from template.");
    } finally {
      setCreatingKey(null);
    }
  }

  const disabled = pending || creatingKey !== null;

  return (
    <div className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Step 1 of 3
        </p>
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Start from a template
        </h1>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">
          Pick a ready-made schema — you can still customize every field in the next step.
        </p>
      </div>

      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          {error}
        </p>
      ) : null}

      {templates === null ? (
        <p className="text-sm text-muted-foreground">Loading templates…</p>
      ) : (
        <ul className="flex flex-col gap-3" aria-label="Starter templates">
          {templates.map((template) => (
            <li key={template.key}>
              <button
                type="button"
                onClick={() => handleSelect(template)}
                disabled={disabled}
                className="w-full rounded-xl border border-border bg-background px-4 py-3 text-left transition-colors hover:border-terracotta hover:bg-muted disabled:pointer-events-none disabled:opacity-50"
              >
                <p className="font-medium">
                  {template.name}
                  {creatingKey === template.key ? (
                    <span className="ml-2 text-xs text-muted-foreground">Creating…</span>
                  ) : null}
                </p>
                <p className="mt-1 text-sm text-muted-foreground text-pretty">
                  {template.description}
                </p>
                <p className="mt-2 font-[family-name:var(--font-data)] text-xs tracking-wide text-sage uppercase">
                  {template.category} · {template.default_target_rows.toLocaleString()} rows
                </p>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
