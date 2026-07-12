/**
 * Compact 1-2-3 progress rail for the wizard. Purely presentational: the
 * caller decides which step is "current" (and therefore which are "done"
 * vs "upcoming").
 */
export interface WizardStepMeta {
  key: string;
  label: string;
}

export function StepIndicator({
  steps,
  current,
}: {
  steps: WizardStepMeta[];
  current: string;
}) {
  const currentIndex = steps.findIndex((s) => s.key === current);

  return (
    <ol className="flex items-center gap-2 sm:gap-4" aria-label="Wizard progress">
      {steps.map((s, i) => {
        const state = i < currentIndex ? "done" : i === currentIndex ? "current" : "upcoming";
        return (
          <li key={s.key} className="flex flex-1 items-center gap-2 sm:gap-4 last:flex-none">
            <span
              aria-current={state === "current" ? "step" : undefined}
              className={
                "flex size-8 shrink-0 items-center justify-center rounded-full border font-[family-name:var(--font-data)] text-xs font-medium transition-colors duration-300 " +
                (state === "current"
                  ? "border-terracotta bg-terracotta text-terracotta-foreground"
                  : state === "done"
                    ? "border-sage bg-sage text-sage-foreground"
                    : "border-border bg-background text-muted-foreground")
              }
            >
              {i + 1}
            </span>
            <span
              className={
                "hidden text-sm font-medium sm:inline " +
                (state === "upcoming" ? "text-muted-foreground" : "text-foreground")
              }
            >
              {s.label}
            </span>
            {i < steps.length - 1 ? (
              <span
                aria-hidden
                className={
                  "h-px flex-1 transition-colors duration-300 " +
                  (state === "done" ? "bg-sage" : "bg-border")
                }
              />
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}
