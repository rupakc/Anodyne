"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { createApiClient, type ApiClient, type DatasetSpec, type FieldSpec } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Field, TextInput, TextArea } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import { StepIndicator, type WizardStepMeta } from "./step-indicator";
import { ReviewStep } from "./review-step";
import { ConfirmStep } from "./confirm-step";

type Step = "describe" | "review" | "confirm";

const STEPS: WizardStepMeta[] = [
  { key: "describe", label: "Task" },
  { key: "review", label: "Review schema" },
  { key: "confirm", label: "Generate" },
];

const TASK_TYPES = [
  { key: "classification", label: "Classification", hint: "Records have a text field and a label." },
  { key: "qa", label: "Question answering", hint: "Records have a question and an answer." },
  { key: "summarization", label: "Summarization", hint: "Records have a document and a summary." },
  { key: "chat", label: "Chat / instruction", hint: "Records have an instruction and a response." },
  { key: "generic", label: "Generic text", hint: "Free-form text records." },
] as const;

type TaskKey = (typeof TASK_TYPES)[number]["key"];

const TASK_PROMPT: Record<TaskKey, string> = {
  classification:
    "A text-classification corpus: each record has a free-text `text` field and a categorical `label`.",
  qa: "A question-answering corpus: each record has a `question` and its `answer`.",
  summarization: "A summarization corpus: each record has a long `document` and a short `summary`.",
  chat: "A chat/instruction corpus: each record has an `instruction` and a model `response`.",
  generic: "A generic text corpus of free-form documents.",
};

export interface TextWizardProps {
  accessToken?: string;
  api?: ApiClient;
}

/**
 * Text-corpus generation. Mirrors the tabular schema flow (describe → review
 * → confirm → generate) but seeds the description from the chosen task type
 * (classification/QA/summarization/chat) and creates the dataset with
 * `modality: "text"`, so the gateway proposes the matching record shape.
 */
export function TextWizard({ accessToken, api: injectedApi }: TextWizardProps) {
  const router = useRouter();
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);

  const [step, setStep] = useState<Step>("describe");
  const [name, setName] = useState("");
  const [task, setTask] = useState<TaskKey>("classification");
  const [description, setDescription] = useState("");
  const [targetRows, setTargetRows] = useState(1000);
  const [spec, setSpec] = useState<DatasetSpec | null>(null);
  const [fields, setFields] = useState<FieldSpec[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requireReview, setRequireReview] = useState(false);

  const canSubmit = name.trim().length > 0 && targetRows > 0;

  async function handleDescribe(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);
    const composed = [TASK_PROMPT[task], description.trim()].filter(Boolean).join(" ");
    try {
      const created = await api.createDataset({
        name: name.trim(),
        description: composed,
        target_rows: targetRows,
        modality: "text",
      });
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
      const job = await api.generate(spec.id, { require_review: requireReview });
      router.push(`/app/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start generation. Please try again.");
      setPending(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <StepIndicator steps={STEPS} current={step} />

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      {step === "describe" ? (
        <form
          onSubmit={handleDescribe}
          className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
        >
          <div>
            <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Step 1 of 3
            </p>
            <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
              Describe your text corpus
            </h2>
            <p className="mt-2 text-sm text-muted-foreground text-pretty">
              Pick a task type — Anodyne proposes the record shape, and you review it next.
            </p>
          </div>

          <Field label="Dataset name" htmlFor="text-name">
            <TextInput
              id="text-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Support intent classification"
              required
            />
          </Field>

          <fieldset className="flex flex-col gap-2">
            <legend className="mb-1 text-sm font-medium">Task type</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              {TASK_TYPES.map((t) => (
                <label
                  key={t.key}
                  className={
                    "flex cursor-pointer flex-col gap-1 rounded-xl border p-3 text-sm transition-colors " +
                    (task === t.key
                      ? "border-terracotta bg-terracotta/10"
                      : "border-border bg-background hover:bg-muted")
                  }
                >
                  <span className="flex items-center gap-2 font-medium">
                    <input
                      type="radio"
                      name="task-type"
                      value={t.key}
                      checked={task === t.key}
                      onChange={() => setTask(t.key)}
                      className="accent-terracotta"
                    />
                    {t.label}
                  </span>
                  <span className="pl-6 text-xs text-muted-foreground text-pretty">{t.hint}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <Field label="Additional guidance" htmlFor="text-description" hint="Optional — domain, tone, label set, etc.">
            <TextArea
              id="text-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Short customer messages; labels: billing, technical, account."
            />
          </Field>

          <Field label="Target record count" htmlFor="text-target-rows" className="max-w-xs">
            <TextInput
              id="text-target-rows"
              type="number"
              min={1}
              value={targetRows}
              onChange={(e) => setTargetRows(Number(e.target.value) || 0)}
            />
          </Field>

          <div className="flex justify-end">
            <Button type="submit" size="lg" disabled={!canSubmit || pending}>
              {pending ? "Proposing schema…" : "Propose schema"}
            </Button>
          </div>
        </form>
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
          requireReview={requireReview}
          onRequireReviewChange={setRequireReview}
          onBack={() => setStep("review")}
          onGenerate={handleGenerate}
        />
      ) : null}
    </div>
  );
}
