import Link from "next/link";
import { ArrowRight, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

const swatches = [
  { name: "Amber", css: "bg-amber", fg: "text-amber-foreground" },
  { name: "Terracotta", css: "bg-terracotta", fg: "text-terracotta-foreground" },
  { name: "Dusty rose", css: "bg-dusty-rose", fg: "text-dusty-rose-foreground" },
  { name: "Sage", css: "bg-sage", fg: "text-sage-foreground" },
  { name: "Cream", css: "bg-cream border border-border", fg: "text-cream-foreground" },
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
            render={<Link href="#get-started" />}
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
                render={<Link href="#get-started" id="get-started" />}
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

        <section className="px-6 py-16 sm:px-10 sm:py-20">
          <div className="mx-auto max-w-5xl">
            <h2 className="font-[family-name:var(--font-display)] text-xl font-semibold tracking-tight">
              Palette
            </h2>
            <p className="mt-2 max-w-lg text-sm text-muted-foreground">
              Autumn pastels, tuned for readability in both light and dark —
              soft ambers, terracotta, dusty rose, sage, and cream.
            </p>
            <ul className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
              {swatches.map((s) => (
                <li
                  key={s.name}
                  className={`flex h-20 flex-col justify-end rounded-lg p-3 text-xs font-medium ${s.css} ${s.fg}`}
                >
                  {s.name}
                </li>
              ))}
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
