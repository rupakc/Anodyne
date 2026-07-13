# Perturbation

Perturbation is how Anodyne **deliberately roughs up a dataset** so you can see how well your models
and pipelines cope with imperfect, real-world data. Real data is messy — Anodyne lets you add that
mess on purpose, in controlled amounts.

## What you can do today

Take any dataset you've generated and produce a **stressed copy** of it, choosing from several
"families" of damage:

- **Noise** — jitter the values so they're slightly off, the way real measurements drift.
- **Feature drift** — shift a column over time or between groups, mimicking data that changes after
  a model was trained.
- **Outliers / anomalies** — inject rare, extreme records that don't fit the pattern.
- **Bias & edge cases** — nudge the data toward (or away from) certain groups or corner cases you
  want to test against.

These work on **tabular and text** datasets. Knowledge graphs have their own perturbations —
rewiring connections, dropping nodes or edges, and injecting rule-breaking records — see
[Graph Modality](Graph-Modality).

You choose the **family** and an **intensity** (how much to apply). Because every perturbation is
**seeded**, the same settings always produce the same result — so your experiments are repeatable.

Common uses: testing how robust a machine-learning model is, and building deliberately "hard"
datasets to challenge the [Evaluation Engine](Evaluation-Engine).

## Under the hood (in plain terms)

- Perturbation runs as a **reliable background job**, just like generation.
- It never changes your original data. Instead it produces a **new dataset version** that remembers
  which dataset it came from (its "lineage"), so you always keep the clean original alongside the
  stressed one.
- You can then **download** the result in any format ([Export & Storage](Export-and-Storage)) or
  **grade** it with the [Evaluation Engine](Evaluation-Engine).

Everything stays private to your organization — see [Multi-Tenancy & Security](Multi-Tenancy-and-Security).
