"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Field, TextInput, Select } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import {
  PERTURBATION_FAMILIES,
  type ApiClient,
  type PerturbationFamily,
} from "@/lib/api";

const FAMILY_LABEL: Record<PerturbationFamily, string> = {
  noise: "Noise",
  drift: "Drift",
  outliers: "Outliers",
  bias: "Bias",
  edge_case: "Edge case",
};

/**
 * Launch a perturbation on a dataset version: pick a family, intensity,
 * target fields, and seed; `POST .../perturb` returns a job that produces a
 * derived version. On success we navigate to the perturbation progress page.
 */
export function PerturbPanel({
  api,
  datasetId,
  versionId,
  fieldNames,
}: {
  api: ApiClient;
  datasetId: string;
  versionId: string;
  fieldNames: string[];
}) {
  const router = useRouter();
  const [family, setFamily] = useState<PerturbationFamily>("noise");
  const [intensity, setIntensity] = useState(0.2);
  const [seed, setSeed] = useState(0);
  const [targets, setTargets] = useState<string[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleTarget(name: string) {
    setTargets((prev) => (prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]));
  }

  async function handleLaunch() {
    setPending(true);
    setError(null);
    try {
      const job = await api.perturb(datasetId, versionId, {
        family,
        intensity,
        target_fields: targets,
        seed,
      });
      router.push(`/app/perturbations/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch perturbation.");
      setPending(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Family" htmlFor={`perturb-family-${versionId}`}>
          <Select
            id={`perturb-family-${versionId}`}
            value={family}
            onChange={(e) => setFamily(e.target.value as PerturbationFamily)}
          >
            {PERTURBATION_FAMILIES.map((f) => (
              <option key={f} value={f}>
                {FAMILY_LABEL[f]}
              </option>
            ))}
          </Select>
        </Field>

        <Field
          label={`Intensity — ${intensity.toFixed(2)}`}
          htmlFor={`perturb-intensity-${versionId}`}
          hint="0 (subtle) to 1 (severe)."
        >
          <input
            id={`perturb-intensity-${versionId}`}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={intensity}
            onChange={(e) => setIntensity(Number(e.target.value))}
            className="accent-terracotta"
          />
        </Field>
      </div>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-sm font-medium">Target fields</legend>
        {fieldNames.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No field metadata available — the whole version will be perturbed.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {fieldNames.map((name) => {
              const on = targets.includes(name);
              return (
                <button
                  key={name}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleTarget(name)}
                  className={
                    "rounded-full border px-3 py-1 font-[family-name:var(--font-data)] text-xs transition-colors " +
                    (on
                      ? "border-terracotta bg-terracotta/15 text-foreground"
                      : "border-border bg-background text-muted-foreground hover:bg-muted")
                  }
                >
                  {name}
                </button>
              );
            })}
          </div>
        )}
        <p className="text-xs text-muted-foreground">
          {targets.length === 0 ? "None selected → all fields." : `${targets.length} selected.`}
        </p>
      </fieldset>

      <Field label="Seed" htmlFor={`perturb-seed-${versionId}`} className="max-w-[10rem]">
        <TextInput
          id={`perturb-seed-${versionId}`}
          type="number"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value) || 0)}
        />
      </Field>

      <div className="flex justify-end">
        <Button type="button" onClick={handleLaunch} disabled={pending}>
          {pending ? "Launching…" : "Launch perturbation"}
        </Button>
      </div>
    </div>
  );
}
