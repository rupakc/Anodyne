"use client";

import { useEffect, useState, type FormEvent } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, TextInput } from "@/components/ui/form";
import { ErrorAlert, Loading } from "@/components/ui/feedback";
import { ApiError, type Annotation, type ApiClient } from "@/lib/api";

type LoadState = "loading" | "ready" | "unavailable" | "error";

/**
 * Row/record annotations for one version (`GET/POST .../annotations`,
 * `DELETE /annotations/{id}`). Tolerates the HITL routes being absent: a 404
 * on load shows an "unavailable" note instead of an error.
 */
export function AnnotationsPanel({
  api,
  datasetId,
  versionId,
}: {
  api: ApiClient;
  datasetId: string;
  versionId: string;
}) {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  const [rowIndex, setRowIndex] = useState("");
  const [label, setLabel] = useState("");
  const [tags, setTags] = useState("");
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.listAnnotations(datasetId, versionId).then(
      (list) => {
        if (!cancelled) {
          setAnnotations(list);
          setState("ready");
        }
      },
      (err) => {
        if (cancelled) return;
        if (err instanceof ApiError && (err.status === 404 || err.status === 501)) {
          setState("unavailable");
        } else {
          setError(err instanceof Error ? err.message : "Failed to load annotations.");
          setState("error");
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, datasetId, versionId]);

  async function handleAdd(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const created = await api.createAnnotation(datasetId, versionId, {
        row_index: rowIndex ? Number(rowIndex) : undefined,
        label: label.trim() || undefined,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        comment: comment.trim() || undefined,
      });
      setAnnotations((prev) => [created, ...prev]);
      setRowIndex("");
      setLabel("");
      setTags("");
      setComment("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save annotation.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await api.deleteAnnotation(id);
      setAnnotations((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete annotation.");
    }
  }

  if (state === "loading") return <Loading label="Loading annotations…" />;
  if (state === "unavailable")
    return (
      <p className="rounded-xl border border-dashed border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
        Annotations aren&apos;t enabled for this tenant yet.
      </p>
    );

  return (
    <div className="flex flex-col gap-4">
      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <form onSubmit={handleAdd} className="flex flex-col gap-3 rounded-xl border border-border bg-background p-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="Row index" htmlFor={`ann-row-${versionId}`} hint="Optional.">
            <TextInput
              id={`ann-row-${versionId}`}
              type="number"
              min={0}
              value={rowIndex}
              onChange={(e) => setRowIndex(e.target.value)}
              placeholder="e.g. 42"
            />
          </Field>
          <Field label="Label" htmlFor={`ann-label-${versionId}`}>
            <TextInput id={`ann-label-${versionId}`} value={label} onChange={(e) => setLabel(e.target.value)} />
          </Field>
          <Field label="Tags" htmlFor={`ann-tags-${versionId}`} hint="Comma-separated.">
            <TextInput id={`ann-tags-${versionId}`} value={tags} onChange={(e) => setTags(e.target.value)} />
          </Field>
        </div>
        <Field label="Comment" htmlFor={`ann-comment-${versionId}`}>
          <TextInput
            id={`ann-comment-${versionId}`}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What did you notice?"
          />
        </Field>
        <div className="flex justify-end">
          <Button type="submit" size="sm" disabled={saving}>
            {saving ? "Saving…" : "Add annotation"}
          </Button>
        </div>
      </form>

      {annotations.length === 0 ? (
        <p className="text-sm text-muted-foreground">No annotations yet.</p>
      ) : (
        <ul className="flex flex-col gap-2" aria-label="Annotations">
          {annotations.map((a) => (
            <li
              key={a.id}
              className="flex items-start justify-between gap-3 rounded-xl border border-border bg-background px-4 py-2.5 text-sm"
            >
              <div className="min-w-0">
                <p className="font-medium">
                  {a.label || "Annotation"}
                  {a.row_index != null ? (
                    <span className="ml-2 font-[family-name:var(--font-data)] text-xs text-muted-foreground">
                      row {a.row_index}
                    </span>
                  ) : null}
                </p>
                {a.comment ? <p className="mt-0.5 text-muted-foreground text-pretty">{a.comment}</p> : null}
                {a.tags && a.tags.length > 0 ? (
                  <p className="mt-1 flex flex-wrap gap-1">
                    {a.tags.map((t) => (
                      <span key={t} className="rounded-full bg-secondary px-2 py-0.5 text-xs">
                        {t}
                      </span>
                    ))}
                  </p>
                ) : null}
              </div>
              <button
                type="button"
                aria-label="Delete annotation"
                onClick={() => handleDelete(a.id)}
                className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              >
                <Trash2 className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
