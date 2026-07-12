"use client";

import { Button } from "@/components/ui/button";
import { SEMANTIC_TYPES, type DatasetSpec, type FieldSpec, type SemanticType } from "@/lib/api";

/**
 * Table view over the proposed `fields`, editable in place. Field order and
 * count come straight from the LLM proposal (adding/removing fields is out
 * of scope here) — this step only lets a reviewer correct what came back:
 * rename a field, re-pick its semantic type, toggle nullability, or tweak
 * its constraints (kept as raw JSON since `FieldSpec.constraints` is an
 * open `dict[str, object]` on the backend).
 */
export function ReviewStep({
  spec,
  fields,
  pending,
  onChange,
  onBack,
  onNext,
}: {
  spec: DatasetSpec;
  fields: FieldSpec[];
  pending: boolean;
  onChange: (fields: FieldSpec[]) => void;
  onBack: () => void;
  onNext: () => void;
}) {
  function updateField(index: number, patch: Partial<FieldSpec>) {
    onChange(fields.map((f, i) => (i === index ? { ...f, ...patch } : f)));
  }

  function updateConstraints(index: number, raw: string) {
    try {
      const parsed = raw.trim() === "" ? {} : JSON.parse(raw);
      updateField(index, { constraints: parsed as Record<string, unknown> });
    } catch {
      // Invalid JSON mid-edit is expected while typing; leave the last
      // valid constraints value in state rather than throwing.
    }
  }

  return (
    <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
      <div>
        <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Step 2 of 3
        </p>
        <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
          Review the proposed schema
        </h2>
        <p className="mt-2 text-sm text-muted-foreground text-pretty">
          &ldquo;{spec.name}&rdquo; — {fields.length} field{fields.length === 1 ? "" : "s"} proposed
          for {spec.target_rows.toLocaleString()} rows. Edit anything before generating.
        </p>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-left text-xs tracking-wide text-muted-foreground uppercase">
              <th className="px-4 py-3 font-medium">Field name</th>
              <th className="px-4 py-3 font-medium">Semantic type</th>
              <th className="px-4 py-3 font-medium">Nullable</th>
              <th className="px-4 py-3 font-medium">Constraints (JSON)</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((field, index) => (
              <tr key={index} className="border-b border-border last:border-0 even:bg-muted/40">
                <td className="px-4 py-2">
                  <input
                    className="w-full rounded-md border border-input bg-background px-2 py-1.5 font-[family-name:var(--font-data)] text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50"
                    value={field.name}
                    onChange={(e) => updateField(index, { name: e.target.value })}
                    aria-label={`Field ${index + 1} name`}
                  />
                </td>
                <td className="px-4 py-2">
                  <select
                    className="w-full rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50"
                    value={field.semantic_type}
                    onChange={(e) =>
                      updateField(index, { semantic_type: e.target.value as SemanticType })
                    }
                    aria-label={`Field ${index + 1} semantic type`}
                  >
                    {SEMANTIC_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="px-4 py-2 text-center">
                  <input
                    type="checkbox"
                    className="size-4 accent-terracotta"
                    checked={field.nullable}
                    onChange={(e) => updateField(index, { nullable: e.target.checked })}
                    aria-label={`Field ${index + 1} nullable`}
                  />
                </td>
                <td className="px-4 py-2">
                  <input
                    className="w-full rounded-md border border-input bg-background px-2 py-1.5 font-[family-name:var(--font-data)] text-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/50"
                    defaultValue={JSON.stringify(field.constraints)}
                    onBlur={(e) => updateConstraints(index, e.target.value)}
                    aria-label={`Field ${index + 1} constraints`}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex justify-between">
        <Button type="button" variant="outline" onClick={onBack} disabled={pending}>
          Back
        </Button>
        <Button type="button" size="lg" onClick={onNext} disabled={pending || fields.length === 0}>
          {pending ? "Saving…" : "Save & continue"}
        </Button>
      </div>
    </section>
  );
}
