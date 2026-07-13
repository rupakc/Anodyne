"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { createApiClient, type ApiClient, type Modality, type ProviderKind } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Field, TextInput, TextArea } from "@/components/ui/form";
import { ErrorAlert } from "@/components/ui/feedback";
import { ProviderSelect } from "@/components/provider-select";

type MediaModality = Extract<Modality, "image" | "audio" | "video">;

const PROVIDER_KIND: Record<MediaModality, ProviderKind> = {
  image: "image-providers",
  audio: "audio-providers",
  video: "video-providers",
};

const COPY: Record<MediaModality, { title: string; blurb: string; countLabel: string }> = {
  image: {
    title: "Generate an image dataset",
    blurb: "Produce labelled synthetic images from prompts using a registered image provider.",
    countLabel: "Number of images",
  },
  audio: {
    title: "Generate an audio dataset",
    blurb: "Synthesize speech/audio clips from prompts using a registered audio provider.",
    countLabel: "Number of clips",
  },
  video: {
    title: "Generate a video dataset",
    blurb: "Preview: synthesize short video clips from prompts using a registered video provider.",
    countLabel: "Number of clips",
  },
};

export interface MediaWizardProps {
  modality: MediaModality;
  accessToken?: string;
  api?: ApiClient;
}

/**
 * Create-and-generate flow for the media modalities (image/audio/video).
 * Unlike the tabular/text schema flow there's no field review step — the
 * dataset is created and generation is kicked off in one go, gated on a
 * registered provider for the modality (ProviderSelect surfaces the
 * "no provider / GPU configured" case).
 */
export function MediaWizard({ modality, accessToken, api: injectedApi }: MediaWizardProps) {
  const router = useRouter();
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const copy = COPY[modality];

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [count, setCount] = useState(50);
  const [labels, setLabels] = useState("");
  const [voice, setVoice] = useState("");
  const [language, setLanguage] = useState("");
  const [providerId, setProviderId] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const labelList = labels
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  const canSubmit = name.trim().length > 0 && count > 0 && providerId.length > 0;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);
    try {
      let datasetId: string;
      if (modality === "image") {
        const ds = await api.createImageDataset({
          name: name.trim(),
          description: description.trim(),
          target_count: count,
          labels: labelList,
          directives: { model_config_id: providerId },
        });
        datasetId = ds.id;
      } else if (modality === "audio") {
        const ds = await api.createAudioDataset({
          name: name.trim(),
          description: description.trim(),
          target_rows: count,
          directives: {
            model_config_id: providerId,
            labels: labelList.length ? labelList : undefined,
            voice: voice.trim() || undefined,
            language: language.trim() || undefined,
          },
        });
        datasetId = ds.id;
      } else {
        // Video has no dedicated create route yet — use the generic dataset
        // route with modality="video" and let the backend validate.
        const ds = await api.createDataset({
          name: name.trim(),
          description: description.trim(),
          target_rows: count,
          modality: "video",
        });
        datasetId = ds.id;
      }

      const job = await api.generate(datasetId, { model_config_id: providerId });
      router.push(`/app/jobs/${job.id}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Failed to start ${modality} generation. Please try again.`,
      );
      setPending(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
    >
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          {modality} generation
        </p>
        <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          {copy.title}
        </h2>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">{copy.blurb}</p>
      </div>

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <ProviderSelect
        api={api}
        kind={PROVIDER_KIND[modality]}
        label={`${modality} provider`}
        value={providerId}
        onChange={setProviderId}
        id="media-provider"
      />

      <Field label="Dataset name" htmlFor="media-name">
        <TextInput
          id="media-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Product photos"
          required
        />
      </Field>

      <Field label="Prompt / description" htmlFor="media-description">
        <TextArea
          id="media-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what each item should depict…"
        />
      </Field>

      <Field
        label="Labels"
        htmlFor="media-labels"
        hint="Optional, comma-separated — used to condition/organize the output."
      >
        <TextInput
          id="media-labels"
          value={labels}
          onChange={(e) => setLabels(e.target.value)}
          placeholder="cat, dog, bird"
        />
      </Field>

      {modality === "audio" ? (
        <div className="grid gap-6 sm:grid-cols-2">
          <Field label="Voice" htmlFor="media-voice" hint="Optional.">
            <TextInput id="media-voice" value={voice} onChange={(e) => setVoice(e.target.value)} placeholder="alloy" />
          </Field>
          <Field label="Language" htmlFor="media-language" hint="Optional (BCP-47).">
            <TextInput
              id="media-language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              placeholder="en-US"
            />
          </Field>
        </div>
      ) : null}

      <Field label={copy.countLabel} htmlFor="media-count" className="max-w-xs">
        <TextInput
          id="media-count"
          type="number"
          min={1}
          value={count}
          onChange={(e) => setCount(Number(e.target.value) || 0)}
        />
      </Field>

      <div className="flex justify-end">
        <Button type="submit" size="lg" disabled={!canSubmit || pending}>
          {pending ? "Starting generation…" : "Generate"}
        </Button>
      </div>
    </form>
  );
}
