# Export & Storage

Once Anodyne has made a dataset, you'll want to **take it with you**. Export & Storage is the part
that hands you the finished data as a file, in whatever format your tools expect.

## What you can do today

Download any dataset version in the format you choose:

- **CSV** — opens in Excel, Google Sheets, and almost everything.
- **JSON (lines)** — one record per line, ideal for scripts and data pipelines.
- **Parquet** — compact and fast, the go-to for large datasets and data engineering.
- **Arrow** — for high-performance, in-memory analytics tools.

If you don't pick a format, Anodyne chooses a sensible default — **very large datasets default to
Parquet** so downloads stay small and quick. Export works across **every modality**, including text
datasets (and knowledge graphs have their own rich set of formats — see
[Graph Modality](Graph-Modality)).

Files download with **proper, recognizable names** (like `customers.csv`), so you always know what
you got.

## Under the hood (in plain terms)

- Downloads **stream straight through the app** while you're signed in. There are **no temporary
  links that expire** — earlier "signature expired" and "failed to fetch" download errors are gone
  for good.
- Large datasets are converted in **small batches**, so even huge files export without straining
  memory.
- Your data lives in secure, per-company storage; only your organization can reach it — see
  [Multi-Tenancy & Security](Multi-Tenancy-and-Security).

From here you might **stress-test** the data ([Perturbation](Perturbation)) or **grade** it
([Evaluation Engine](Evaluation-Engine)).
