import Link from "next/link";
import {
  ArrowRight,
  BookOpen,
  Table2,
  FileText,
  Image as ImageIcon,
  AudioLines,
  Video,
  Share2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

// Supported generation modalities, each keyed to a brand hue from the
// autumn-pastel palette (bg/fg tokens defined in globals.css) so the section
// stays on-theme in both light and dark.
const modalities = [
  {
    name: "Tabular",
    body: "Typed columns from a description or a sample — Gaussian copula, CTGAN, TVAE.",
    icon: Table2,
    css: "bg-amber",
    fg: "text-amber-foreground",
  },
  {
    name: "Text",
    body: "Labelled corpora for classification, QA, summarization, and chat.",
    icon: FileText,
    css: "bg-terracotta",
    fg: "text-terracotta-foreground",
  },
  {
    name: "Image",
    body: "Provider-agnostic image sets, from SDXL or an external API.",
    icon: ImageIcon,
    css: "bg-dusty-rose",
    fg: "text-dusty-rose-foreground",
  },
  {
    name: "Audio",
    body: "Speech and audio clips via self-hosted or hosted text-to-speech.",
    icon: AudioLines,
    css: "bg-sage",
    fg: "text-sage-foreground",
  },
  {
    name: "Video",
    body: "Short text-to-video clips, provider-agnostic.",
    icon: Video,
    css: "bg-cream border border-border",
    fg: "text-cream-foreground",
  },
  {
    name: "Graph",
    body: "Knowledge graphs & ontologies — RDF/OWL, property-graph, and GNN exports.",
    icon: Share2,
    css: "bg-amber",
    fg: "text-amber-foreground",
  },
] as const;

const steps = [
  {
    n: "01",
    title: "Describe",
    body: "Write what the dataset should look like in plain English — fields, ranges, relationships.",
  },
  {
    n: "02",
    title: "Review the schema",
    body: "Anodyne proposes a typed schema from your description. Adjust it before a single row is generated.",
  },
  {
    n: "03",
    title: "Generate & download",
    body: "Rows are generated deterministically from a seed and land as a versioned, downloadable dataset.",
  },
] as const;

export default function Home() {
  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center justify-between border-b border-border px-6 py-4 sm:px-10">
        <span className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
          anodyne
        </span>
        <nav className="flex items-center gap-2 sm:gap-4">
          <Link
            href="#how-it-works"
            className="hidden text-sm text-muted-foreground transition-colors hover:text-foreground sm:inline-block"
          >
            How it works
          </Link>
          <ThemeToggle />
          <Button
            render={<Link href="/login" />}
            nativeButton={false}
            size="sm"
          >
            Sign in
            <ArrowRight className="size-3.5" data-icon="inline-end" />
          </Button>
        </nav>
      </header>

      <main className="flex-1">
        <section className="relative overflow-hidden px-6 py-20 sm:px-10 sm:py-28">
          <div
            aria-hidden
            className="grain-motif pointer-events-none absolute inset-y-0 right-0 hidden w-1/2 [mask-image:linear-gradient(to_left,black,transparent)] md:block"
          />
          <div className="relative mx-auto max-w-3xl">
            <p className="mb-4 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Synthetic data, described — not queried
            </p>
            <h1 className="font-[family-name:var(--font-display)] text-4xl leading-[1.08] font-semibold tracking-tight text-balance sm:text-6xl">
              Describe a dataset. Generate the real shape of it.
            </h1>
            <p className="mt-6 max-w-xl text-lg text-muted-foreground text-pretty">
              No source data required. Tell Anodyne what a table should
              contain, review the schema it proposes, and get a versioned,
              downloadable dataset generated to match — reproducibly, from a
              seed.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Button
                render={<Link href="/login" />}
                nativeButton={false}
                size="lg"
              >
                Get started
                <ArrowRight className="size-4" data-icon="inline-end" />
              </Button>
              <Button
                render={<Link href="#how-it-works" />}
                nativeButton={false}
                variant="outline"
                size="lg"
              >
                <BookOpen className="size-4" data-icon="inline-start" />
                View docs
              </Button>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="border-t border-border bg-card px-6 py-16 sm:px-10 sm:py-20">
          <div className="mx-auto max-w-5xl">
            <h2 className="font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight sm:text-3xl">
              Three steps, in order
            </h2>
            <div className="mt-10 grid gap-8 sm:grid-cols-3">
              {steps.map((step) => (
                <div key={step.n} className="flex flex-col gap-3">
                  <span className="font-[family-name:var(--font-data)] text-sm text-terracotta">
                    {step.n}
                  </span>
                  <h3 className="text-lg font-semibold">{step.title}</h3>
                  <p className="text-sm text-muted-foreground text-pretty">
                    {step.body}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="modalities" className="px-6 py-16 sm:px-10 sm:py-20">
          <div className="mx-auto max-w-5xl">
            <h2 className="font-[family-name:var(--font-display)] text-xl font-semibold tracking-tight">
              Supported data modalities
            </h2>
            <p className="mt-2 max-w-lg text-sm text-muted-foreground">
              Generate the shape of any dataset — from a description, a
              template, or a sample — across six modalities.
            </p>
            <ul className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {modalities.map((m) => {
                const Icon = m.icon;
                return (
                  <li
                    key={m.name}
                    className="flex flex-col gap-3 rounded-lg border border-border bg-card p-5"
                  >
                    <span
                      className={`flex size-9 items-center justify-center rounded-md ${m.css} ${m.fg}`}
                    >
                      <Icon className="size-4" />
                    </span>
                    <h3 className="text-base font-semibold">{m.name}</h3>
                    <p className="text-sm text-muted-foreground text-pretty">
                      {m.body}
                    </p>
                  </li>
                );
              })}
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-border px-6 py-8 text-sm text-muted-foreground sm:px-10">
        Anodyne — generation engine foundation.
      </footer>
    </div>
  );
}
