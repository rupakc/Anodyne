"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { createApiClient, type ApiClient, type DatasetSpec, type FieldSpec } from "@/lib/api";
import { StepIndicator, type WizardStepMeta } from "./step-indicator";
import { DescribeStep } from "./describe-step";
import { ReviewStep } from "./review-step";
import { ConfirmStep } from "./confirm-step";

type Step = "describe" | "review" | "confirm";

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

      {step === "describe" ? <DescribeStep pending={pending} onSubmit={handleDescribe} /> : null}

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
