# Generation Engine

The Generation Engine is the part of Anodyne that actually **makes datasets**. This page describes
what's available today (the foundation, "C0") and what's coming.

## What you can do today

Create a **table (tabular dataset) from a plain-English description**, entirely through the web app:

1. **Describe it.** e.g. *"A table of online shoppers with age, country, and total spend."* Give it a
   name and how many rows you want.
2. **Review the proposed columns.** Anodyne uses your chosen AI model to suggest the columns (name +
   type + rules, like "age is a whole number between 18 and 90"). You can edit any of them.
3. **Generate.** Anodyne builds the rows. Generation runs as a durable background job that can scale
   across many workers for large datasets.
4. **Watch and download.** A live progress view shows the job running; when it finishes, you download
   the file (Parquet format).

The data is generated to be **reproducible** — the same request with the same settings produces the
same rows — which is important for testing and comparisons.

## What's under the hood (in plain terms)

- A **workflow engine** (Temporal) manages each generation job reliably — it survives restarts and
  can pause for human review.
- A **distributed compute engine** (Ray) does the actual row-building, spreading big jobs across
  workers.
- The **web app** is a friendly, autumn-themed interface for the whole flow.
- Everything is **per-company and secure** — see [Multi-Tenancy & Security](Multi-Tenancy-and-Security).

## What's coming next

The foundation is deliberately a thin, working slice. Planned additions:

- **C1 — Tabular from a sample.** Upload an existing table; Anodyne learns its patterns and generates
  realistic new rows that match.
- **C2 — Text datasets.** Generate text corpora (e.g. for classification, Q&A, chat).
- **C3–C5 — Images, audio, video.** Generate these using open models on your own GPUs or external
  providers.
- **C6 — Templates & targeted cases.** Starter templates plus the ability to deliberately create
  biases, rare edge-cases, and specific scenarios for testing.

## Status

✅ **C0 (foundation) done.** C1–C6 are in progress, built as independent, plug-in modalities on this
foundation.
