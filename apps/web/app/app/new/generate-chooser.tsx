"use client";

import { useState } from "react";
import { ArrowLeft, Table2, FileText, Image as ImageIcon, AudioLines, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SectionHeading } from "@/components/ui/feedback";
import { Wizard } from "./wizard";
import { TextWizard } from "./text-wizard";
import { MediaWizard } from "./media-wizard";

type Flow = "tabular" | "text" | "image" | "audio" | "video";

const MODALITIES: {
  key: Flow;
  label: string;
  blurb: string;
  icon: typeof Table2;
}[] = [
  { key: "tabular", label: "Tabular", blurb: "Describe it, start from a template, or learn the shape from a sample.", icon: Table2 },
  { key: "text", label: "Text corpus", blurb: "Classification, QA, summarization, or chat/instruction data.", icon: FileText },
  { key: "image", label: "Image", blurb: "Labelled synthetic images from a registered image provider.", icon: ImageIcon },
  { key: "audio", label: "Audio", blurb: "Speech/audio clips from a registered audio provider.", icon: AudioLines },
  { key: "video", label: "Video", blurb: "Preview: short clips from a registered video provider.", icon: Video },
];

/**
 * Entry point for `/app/new`: choose a modality, then hand off to the flow
 * that owns it — the tabular {@link Wizard} (describe / sample / template),
 * the {@link TextWizard}, or the {@link MediaWizard} for image/audio/video.
 */
export function GenerateChooser({ accessToken }: { accessToken?: string }) {
  const [flow, setFlow] = useState<Flow | null>(null);

  if (flow === null) {
    return (
      <div className="flex flex-col gap-8">
        <SectionHeading
          eyebrow="Generate"
          title="What do you want to generate?"
          description="Pick a modality to begin. Everything runs through the same review-and-generate workflow."
        />
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MODALITIES.map((m) => (
            <li key={m.key}>
              <button
                type="button"
                onClick={() => setFlow(m.key)}
                className="group flex h-full w-full flex-col gap-3 rounded-2xl border border-border bg-card p-6 text-left shadow-sm transition-colors hover:border-terracotta/50 hover:bg-muted/40 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                <span className="flex size-11 items-center justify-center rounded-xl bg-terracotta/12 text-terracotta transition-colors group-hover:bg-terracotta/20">
                  <m.icon className="size-5" />
                </span>
                <span className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
                  {m.label}
                </span>
                <span className="text-sm text-muted-foreground text-pretty">{m.blurb}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const current = MODALITIES.find((m) => m.key === flow)!;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-4">
        <Button type="button" variant="ghost" size="sm" onClick={() => setFlow(null)}>
          <ArrowLeft className="size-3.5" data-icon="inline-start" />
          All modalities
        </Button>
        <span className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-muted-foreground uppercase">
          {current.label}
        </span>
      </div>

      {flow === "tabular" ? <Wizard accessToken={accessToken} /> : null}
      {flow === "text" ? <TextWizard accessToken={accessToken} /> : null}
      {flow === "image" || flow === "audio" || flow === "video" ? (
        <MediaWizard modality={flow} accessToken={accessToken} />
      ) : null}
    </div>
  );
}
