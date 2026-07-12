"""Report artifacts: a machine-readable JSON blob and a self-contained HTML page.

The HTML is fully inline (no external CSS/JS/font/image URLs) so it renders from
the object store behind a presigned URL with no network dependencies. The
palette is Anodyne's autumn-pastel direction (warm ambers / terracotta / sage /
cream) kept deliberately simple.
"""

from __future__ import annotations

import html

from anodyne_evaluation.models import EvaluationReport

_CSS = """
:root { --cream:#fbf6ee; --ink:#3f3a34; --amber:#d9a05b; --terracotta:#c1666b;
        --sage:#8a9a5b; --rose:#d4a5a5; --card:#fffdf9; --muted:#8c8579; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       background: var(--cream); color: var(--ink); line-height:1.5; }
.wrap { max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 24px; font-size: .9rem; }
.overall { background: var(--card); border-radius: 16px; padding: 24px; margin-bottom: 24px;
           border: 1px solid #efe6d8; box-shadow: 0 2px 10px rgba(160,120,60,.06); }
.score-big { font-size: 3rem; font-weight: 700; color: var(--terracotta); }
.summary { font-size: 1.05rem; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
.card { background: var(--card); border:1px solid #efe6d8; border-radius: 14px; padding: 16px; }
.dim { font-weight: 600; text-transform: capitalize; margin-bottom: 8px; }
.bar { height: 10px; border-radius: 6px; background: #efe6d8; overflow: hidden;
       margin: 6px 0 10px; }
.bar > span { display:block; height:100%;
              background: linear-gradient(90deg, var(--sage), var(--amber)); }
.pct { float: right; color: var(--muted); font-weight: 600; }
.rationale { font-size: .86rem; color: #5c554c; }
.metrics { font-size: .78rem; color: var(--muted); margin-top: 8px; }
.recs { margin-top: 24px; }
.recs li { margin-bottom: 6px; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; margin-top:6px; }
td { padding: 2px 6px; border-top: 1px solid #efe6d8; }
"""


def render_json(report: EvaluationReport) -> bytes:
    """Canonical JSON serialization of the report (round-trips via model_validate)."""
    return report.model_dump_json(indent=2).encode("utf-8")


def _bar(score: float) -> str:
    pct = max(0.0, min(100.0, score * 100.0))
    return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'


def render_html(report: EvaluationReport) -> str:
    cards = []
    for s in sorted(report.expert_scores, key=lambda x: str(x.dimension)):
        metrics_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>" for k, v in s.metrics.items()
        )
        cards.append(
            f'<div class="card"><div class="dim">{html.escape(str(s.dimension))}'
            f'<span class="pct">{s.score * 100:.0f}%</span></div>'
            f"{_bar(s.score)}"
            f'<div class="rationale">{html.escape(s.rationale)}</div>'
            f'<div class="metrics"><table>{metrics_rows}</table></div></div>'
        )
    recs = "".join(f"<li>{html.escape(r)}</li>" for r in report.recommendations)
    recs_block = f'<div class="recs"><h2>Recommendations</h2><ul>{recs}</ul></div>' if recs else ""
    ref = (
        f"vs reference {report.reference_version_id}"
        if report.reference_version_id
        else "no reference (intrinsic metrics only)"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Evaluation Report {report.id}</title><style>{_CSS}</style></head><body>"
        f'<div class="wrap"><h1>360&deg; Evaluation Report</h1>'
        f'<p class="sub">Dataset version {report.dataset_version_id} &mdash; {html.escape(ref)}'
        f" &mdash; {report.created_at:%Y-%m-%d %H:%M UTC}</p>"
        f'<div class="overall"><div class="score-big">{report.overall_score * 100:.0f}'
        '<span style="font-size:1.2rem">/100</span></div>'
        f'<div class="summary">{html.escape(report.summary)}</div></div>'
        f'<div class="grid">{"".join(cards)}</div>{recs_block}</div></body></html>'
    )
