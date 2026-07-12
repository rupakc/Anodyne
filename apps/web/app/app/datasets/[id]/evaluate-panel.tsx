"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Field, TextInput, Select } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import type { ApiClient, DatasetVersion } from "@/lib/api";

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

  async function handleLaunch() {
    setPending(true);
    setError(null);
    try {
      const run = await api.evaluate(datasetId, versionId, {
        reference_version_id: referenceVersionId || undefined,
        sensitive_field: sensitiveField || undefined,
        target_field: targetField || undefined,
        sample_rows: sampleRows,
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

      <div className="flex justify-end">
        <Button type="button" onClick={handleLaunch} disabled={pending}>
          {pending ? "Launching…" : "Launch evaluation"}
        </Button>
      </div>
    </div>
  );
}
