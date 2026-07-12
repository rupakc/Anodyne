"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { createApiClient, type ApiClient, type DatasetSpec, type FieldSpec } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { StepIndicator, type WizardStepMeta } from "./step-indicator";
import { DescribeStep } from "./describe-step";
import { TemplateStep } from "./template-step";
import { SampleStep } from "./sample-step";
import { ReviewStep } from "./review-step";
import { ConfirmStep } from "./confirm-step";

type Step = "describe" | "review" | "confirm";
type Source = "describe" | "sample" | "template";

const STEPS: WizardStepMeta[] = [
  { key: "describe", label: "Describe" },
  { key: "review", label: "Review schema" },
  { key: "confirm", label: "Generate" },
];

export interface WizardProps {
  /** OIDC access token from `session.accessToken` (Task 11). */
  accessToken?: string;
  /**
   * Dependency-injection point: pass a fake/mock `ApiClient` in tests
   * instead of building a real fetch-backed one from `accessToken`.
   */
  api?: ApiClient;
}

/**
 * The create-from-description wizard's state machine. Three steps, driven
 * by the gateway's dataset lifecycle:
 *
 *   describe --createDataset--> review --updateDataset--> confirm --generate--> /app/jobs/{id}
 *
 * Each transition is gated on its own gateway call succeeding; failures
 * surface inline and leave the wizard on the current step so the user can
 * retry without losing their input.
 */
export function Wizard({ accessToken, api: injectedApi }: WizardProps) {
  const router = useRouter();
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);

  const [step, setStep] = useState<Step>("describe");
  const [source, setSource] = useState<Source>("describe");
  const [spec, setSpec] = useState<DatasetSpec | null>(null);
  const [fields, setFields] = useState<FieldSpec[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDescribe(input: { name: string; description: string; target_rows: number }) {
    setPending(true);
    setError(null);
    try {
      const created = await api.createDataset(input);
      setSpec(created);
      setFields(created.fields);
      setStep("review");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to propose a schema. Please try again.");
    } finally {
      setPending(false);
    }
  }

  function handleTemplateCreated(created: DatasetSpec) {
    setSpec(created);
    setFields(created.fields);
    setStep("review");
  }

  function handleSampleCreated(created: DatasetSpec) {
    setSpec(created);
    setFields(created.fields);
    setStep("review");
  }

  async function handleReviewNext() {
    if (!spec) return;
    setPending(true);
    setError(null);
    try {
      const updated = await api.updateDataset(spec.id, { fields });
      setSpec(updated);
      setFields(updated.fields);
      setStep("confirm");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save your edits. Please try again.");
    } finally {
      setPending(false);
    }
  }

  async function handleGenerate() {
    if (!spec) return;
    setPending(true);
    setError(null);
    try {
      const job = await api.generate(spec.id);
      router.push(`/app/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start generation. Please try again.");
      setPending(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <StepIndicator steps={STEPS} current={step} />

      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          {error}
        </p>
      ) : null}

      {step === "describe" ? (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Dataset source">
            <Button
              type="button"
              size="sm"
              variant={source === "describe" ? "default" : "outline"}
              onClick={() => setSource("describe")}
            >
              Describe
            </Button>
            <Button
              type="button"
              size="sm"
              variant={source === "sample" ? "default" : "outline"}
              onClick={() => setSource("sample")}
            >
              From a sample
            </Button>
            <Button
              type="button"
              size="sm"
              variant={source === "template" ? "default" : "outline"}
              onClick={() => setSource("template")}
            >
              Start from a template
            </Button>
          </div>

          {source === "describe" ? (
            <DescribeStep pending={pending} onSubmit={handleDescribe} />
          ) : source === "sample" ? (
            <SampleStep api={api} pending={pending} onCreated={handleSampleCreated} />
          ) : (
            <TemplateStep api={api} pending={pending} onCreated={handleTemplateCreated} />
          )}
        </div>
      ) : null}

      {step === "review" && spec ? (
        <ReviewStep
          spec={spec}
          fields={fields}
          pending={pending}
          onChange={setFields}
          onBack={() => setStep("describe")}
          onNext={handleReviewNext}
        />
      ) : null}

      {step === "confirm" && spec ? (
        <ConfirmStep
          spec={{ ...spec, fields }}
          pending={pending}
          onBack={() => setStep("review")}
          onGenerate={handleGenerate}
        />
      ) : null}
    </div>
  );
}
