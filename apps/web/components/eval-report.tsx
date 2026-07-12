"use client";

import { EVAL_DIMENSIONS, type EvaluationReport, type ExpertScore } from "@/lib/api";
import { FeedbackWidget } from "@/components/feedback-widget";

/** 0..1 → whole-number /100 score used consistently across the report. */
function toHundred(score: number): number {
  return Math.round(Math.max(0, Math.min(1, score)) * 100);
}

/** Nice label for a dimension key (e.g. "edge_case" → "Edge case"). */
function labelFor(dimension: string): string {
  return dimension.charAt(0).toUpperCase() + dimension.slice(1).replace(/_/g, " ");
}

/**
 * Self-contained inline SVG radar chart of the (≤6) dimension scores. No
 * external libraries or CDNs; colors come from the autumn-pastel CSS vars so
 * it stays theme-aware. A hidden text list mirrors the data for screen
 * readers (the <svg> itself is role="img" with a summary aria-label).
 */
function Radar({ scores }: { scores: { dimension: string; score: number }[] }) {
  const size = 260;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 40;
  const n = scores.length;

  const angleFor = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const pointAt = (i: number, radius: number) => ({
    x: cx + radius * Math.cos(angleFor(i)),
    y: cy + radius * Math.sin(angleFor(i)),
  });

  const rings = [0.25, 0.5, 0.75, 1];
  const dataPoints = scores.map((s, i) => pointAt(i, r * Math.max(0, Math.min(1, s.score))));
  const dataPath = dataPoints.map((p) => `${p.x},${p.y}`).join(" ");

  const summary = scores.map((s) => `${labelFor(s.dimension)} ${toHundred(s.score)}`).join(", ");

  if (n < 3) {
    // A radar needs ≥3 axes to read as a shape; the bars below still cover it.
    return null;
  }

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label={`Dimension scores out of 100: ${summary}`}
      className="mx-auto max-w-full text-border"
    >
      {/* grid rings */}
      {rings.map((ring) => (
        <polygon
          key={ring}
          points={scores.map((_, i) => {
            const p = pointAt(i, r * ring);
            return `${p.x},${p.y}`;
          }).join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.5}
        />
      ))}
      {/* spokes */}
      {scores.map((_, i) => {
        const p = pointAt(i, r);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="currentColor" strokeWidth={1} opacity={0.4} />;
      })}
      {/* data polygon */}
      <polygon
        points={dataPath}
        fill="var(--terracotta)"
        fillOpacity={0.22}
        stroke="var(--terracotta)"
        strokeWidth={2}
      />
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill="var(--terracotta)" />
      ))}
      {/* axis labels */}
      {scores.map((s, i) => {
        const p = pointAt(i, r + 20);
        return (
          <text
            key={s.dimension}
            x={p.x}
            y={p.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="var(--muted-foreground)"
            className="font-[family-name:var(--font-data)]"
            fontSize={10}
          >
            {labelFor(s.dimension)}
          </text>
        );
      })}
    </svg>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = toHundred(score);
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      className="h-2 w-full overflow-hidden rounded-full bg-muted"
    >
      <div
        className="h-full rounded-full bg-gradient-to-r from-amber to-terracotta"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function ExpertCard({ expert }: { expert: ExpertScore }) {
  return (
    <li className="flex flex-col gap-3 rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-[family-name:var(--font-display)] text-base font-semibold tracking-tight">
          {labelFor(expert.dimension)}
        </h3>
        <span className="font-[family-name:var(--font-data)] text-sm font-medium text-terracotta">
          {toHundred(expert.score)}
          <span className="text-muted-foreground">/100</span>
        </span>
      </div>
      <ScoreBar score={expert.score} />
      {expert.rationale ? (
        <p className="text-sm text-muted-foreground text-pretty">{expert.rationale}</p>
      ) : null}
      {expert.recommendations.length > 0 ? (
        <ul className="flex list-disc flex-col gap-1 pl-5 text-sm text-muted-foreground">
          {expert.recommendations.map((rec, i) => (
            <li key={i} className="text-pretty">
              {rec}
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

/**
 * Visually-rich, self-contained evaluation report: an overall 360° score, a
 * radar of the six expert dimensions, per-expert cards (score/rationale/
 * recommendations), the qualitative summary, overall recommendations, and a
 * thumbs feedback widget.
 */
export function EvalReport({
  report,
  accessToken,
}: {
  report: EvaluationReport;
  accessToken?: string;
}) {
  // Order experts by the canonical dimension order when possible.
  const experts = [...report.expert_scores].sort(
    (a, b) =>
      EVAL_DIMENSIONS.indexOf(a.dimension as (typeof EVAL_DIMENSIONS)[number]) -
      EVAL_DIMENSIONS.indexOf(b.dimension as (typeof EVAL_DIMENSIONS)[number]),
  );
  const radarScores = experts.map((e) => ({ dimension: e.dimension, score: e.score }));

  return (
    <div className="flex flex-col gap-8">
      <section className="grid gap-6 rounded-2xl border border-border bg-card p-6 shadow-sm sm:grid-cols-[auto_1fr] sm:items-center">
        <div className="flex flex-col items-center gap-1 rounded-2xl bg-gradient-to-br from-amber/15 to-terracotta/15 px-8 py-6 text-center">
          <span className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            360° score
          </span>
          <span className="font-[family-name:var(--font-display)] text-5xl font-semibold tracking-tight">
            {toHundred(report.overall_score)}
          </span>
          <span className="text-xs text-muted-foreground">out of 100</span>
        </div>
        <div>
          <Radar scores={radarScores} />
          {/* a11y / no-JS text fallback */}
          <ul className="sr-only">
            {radarScores.map((s) => (
              <li key={s.dimension}>
                {labelFor(s.dimension)}: {toHundred(s.score)} out of 100
              </li>
            ))}
          </ul>
        </div>
      </section>

      {report.summary ? (
        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm">
          <h2 className="mb-2 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Summary
          </h2>
          <p className="text-sm text-muted-foreground text-pretty">{report.summary}</p>
        </section>
      ) : null}

      <section>
        <h2 className="mb-3 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
          Per-expert scores
        </h2>
        <ul className="grid gap-3 sm:grid-cols-2" aria-label="Expert scores">
          {experts.map((e) => (
            <ExpertCard key={e.dimension} expert={e} />
          ))}
        </ul>
      </section>

      {report.recommendations.length > 0 ? (
        <section className="rounded-2xl border border-sage/40 bg-sage/10 p-6">
          <h2 className="mb-2 font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
            Recommendations
          </h2>
          <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-foreground">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="text-pretty">
                {rec}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <FeedbackWidget
        targetType="evaluation_report"
        targetId={report.id}
        accessToken={accessToken}
        label="Rate this evaluation"
      />
    </div>
  );
}
