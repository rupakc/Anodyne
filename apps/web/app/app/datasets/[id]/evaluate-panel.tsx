"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Field, TextInput, Select } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import type { ApiClient, DatasetVersion, TaskMetricsCatalog } from "@/lib/api";

/** Nice label for a task-class key (e.g. "text_classification" → "Text classification"). */
function humanizeTaskType(taskType: string): string {
  return taskType.charAt(0).toUpperCase() + taskType.slice(1).replace(/_/g, " ");
}

/**
 * Launch an LLM-as-a-Judge evaluation on a version. An optional reference
 * version anchors fidelity/privacy comparisons; optional sensitive/target
 * fields sharpen the privacy/utility experts. On success we navigate to the
 * evaluation run page, which polls and renders the 360° report.
 */
export function EvaluatePanel({
  api,
  datasetId,
  versionId,
  otherVersions,
  fieldNames,
}: {
  api: ApiClient;
  datasetId: string;
  versionId: string;
  otherVersions: DatasetVersion[];
  fieldNames: string[];
}) {
  const router = useRouter();
  const [referenceVersionId, setReferenceVersionId] = useState("");
  const [sensitiveField, setSensitiveField] = useState("");
  const [targetField, setTargetField] = useState("");
  const [sampleRows, setSampleRows] = useState(20);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Per-task standard metrics catalog (`GET .../task-metrics`). Absent/empty
  // catalogs (including a fetch failure) hide the panel entirely — the
  // evaluation still launches fine, just with the gateway's full default set.
  const [taskMetrics, setTaskMetrics] = useState<TaskMetricsCatalog | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    api.fetchTaskMetrics(datasetId, versionId).then(
      (catalog) => {
        if (cancelled) return;
        setTaskMetrics(catalog);
        setSelectedMetrics(new Set(catalog.available_metrics.map((m) => m.key)));
      },
      () => {
        // Fetch failed (route absent/404, generic task with no catalog, etc.)
        // — degrade gracefully, never block launching the evaluation.
        if (!cancelled) {
          setTaskMetrics(null);
          setSelectedMetrics(new Set());
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, datasetId, versionId]);

  const availableMetrics = taskMetrics?.available_metrics ?? [];

  function toggleMetric(key: string) {
    setSelectedMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  async function handleLaunch() {
    setPending(true);
    setError(null);
    try {
      const run = await api.evaluate(datasetId, versionId, {
        reference_version_id: referenceVersionId || undefined,
        sensitive_field: sensitiveField || undefined,
        target_field: targetField || undefined,
        sample_rows: sampleRows,
        selected_metrics:
          availableMetrics.length > 0
            ? availableMetrics.map((m) => m.key).filter((key) => selectedMetrics.has(key))
            : undefined,
      });
      router.push(`/app/evaluations/${run.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch evaluation.");
      setPending(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Reference version"
          htmlFor={`eval-ref-${versionId}`}
          hint="Optional — compare against a real/baseline version."
        >
          <Select
            id={`eval-ref-${versionId}`}
            value={referenceVersionId}
            onChange={(e) => setReferenceVersionId(e.target.value)}
          >
            <option value="">None</option>
            {otherVersions.map((v) => (
              <option key={v.id} value={v.id}>
                {v.format.toUpperCase()} · {v.row_count.toLocaleString()} rows ·{" "}
                {new Date(v.created_at).toLocaleDateString()}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Sample rows" htmlFor={`eval-sample-${versionId}`} hint="Rows sampled per expert.">
          <TextInput
            id={`eval-sample-${versionId}`}
            type="number"
            min={1}
            value={sampleRows}
            onChange={(e) => setSampleRows(Number(e.target.value) || 0)}
          />
        </Field>

        <Field label="Sensitive field" htmlFor={`eval-sensitive-${versionId}`} hint="Optional — for the privacy expert.">
          <Select
            id={`eval-sensitive-${versionId}`}
            value={sensitiveField}
            onChange={(e) => setSensitiveField(e.target.value)}
          >
            <option value="">None</option>
            {fieldNames.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Target field" htmlFor={`eval-target-${versionId}`} hint="Optional — for the utility expert.">
          <Select
            id={`eval-target-${versionId}`}
            value={targetField}
            onChange={(e) => setTargetField(e.target.value)}
          >
            <option value="">None</option>
            {fieldNames.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Field>
      </div>

      {availableMetrics.length > 0 ? (
        <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
          <div>
            <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Standard metrics
            </p>
            <h3 className="mt-1 text-sm font-semibold text-foreground">
              {humanizeTaskType(taskMetrics!.task_type)}
            </h3>
          </div>
          <ul className="flex flex-col gap-2">
            {availableMetrics.map((m) => {
              const inputId = `eval-metric-${versionId}-${m.key}`;
              return (
                <li
                  key={m.key}
                  className="flex items-start gap-3 rounded-lg border border-border bg-background px-3 py-2"
                >
                  <input
                    id={inputId}
                    type="checkbox"
                    className="mt-0.5 size-4 accent-terracotta"
                    checked={selectedMetrics.has(m.key)}
                    onChange={() => toggleMetric(m.key)}
                  />
                  <label htmlFor={inputId} className="flex flex-col gap-0.5 text-sm">
                    <span className="font-medium text-foreground">
                      {m.label}{" "}
                      <span className="font-[family-name:var(--font-data)] text-xs font-normal text-muted-foreground">
                        {m.key}
                      </span>
                    </span>
                    {m.description ? (
                      <span className="text-xs text-muted-foreground text-pretty">{m.description}</span>
                    ) : null}
                  </label>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}

      <div className="flex justify-end">
        <Button type="button" onClick={handleLaunch} disabled={pending}>
          {pending ? "Launching…" : "Launch evaluation"}
        </Button>
      </div>
    </div>
  );
}
