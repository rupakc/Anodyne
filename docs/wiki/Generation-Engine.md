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

## All five modalities are now available

Beyond the tabular-from-description flow above, Anodyne now generates every modality:

- **Tabular from a sample.** Upload an existing table; Anodyne learns its patterns (distributions,
  correlations) and generates realistic new rows that match — using privacy-safe statistical models
  (with a high-fidelity option). Personal-looking fields (names, emails) are always faked, never
  copied from your sample.
- **Text datasets.** Generate text corpora for classification, Q&A, summarization, and chat — with
  automatic de-duplication and quality filtering.
- **Images, audio, and video.** Generate these through your choice of provider — open models on your
  own GPUs, or external services — using per-company API keys (kept encrypted).
- **Starter templates & targeted cases.** Pick a ready-made template (customers, transactions,
  support tickets, sensor readings, users-with-churn) and customize it, and deliberately steer the
  data toward specific biases, rare edge-cases, or use-cases for testing.

## Status

✅ **Generation Engine complete (C0–C6).** All five modalities plus templates and targeted-case
steering are built, behind one consistent interface. Live image/audio/video generation requires a
GPU or an external provider key; the rest runs on an ordinary machine (offline for tabular/text).
