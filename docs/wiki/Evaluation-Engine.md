# Evaluation Engine

Making synthetic data is only half the job — you also need to know **how good it is**. The
Evaluation Engine is Anodyne's **panel of AI expert judges** that grades a dataset from every angle
and gives you a clear report.

## What you can do today

Point the Evaluation Engine at any dataset version and get back a **360° quality review**. A
mixture of specialist "experts" each scores a different dimension:

- **Fidelity** — does the synthetic data look statistically like real data should?
- **Diversity** — is there enough variety, or is it repetitive?
- **Privacy** — does it avoid leaking or memorizing anything sensitive?
- **Utility** — is it actually useful? (Tested by training a model on the synthetic data and
  checking how it does on real data.)
- **Bias** — are groups represented fairly?
- **Qualitative judgment** — an AI reviewer reads samples and comments in plain language.

You get an **overall score** plus a **per-expert breakdown**, delivered two ways: as a **data file
(JSON)** for your own tooling, and as a **visual HTML report** with a radar chart. In the web app,
each expert's card is **clickable** to see the detail behind its score.

Knowledge graphs are judged by their own specialist experts — structural fidelity, ontology
consistency, connectivity, privacy, usefulness for graph learning, and semantic plausibility. See
[Graph Modality](Graph-Modality).

## Under the hood (in plain terms)

- Evaluation runs as a **reliable background job**; you watch progress and open the report when it's
  done.
- The qualitative judge uses whichever AI model your organization has configured — see
  [Bring Your Own AI Model](LLM-Abstraction).
- The report viewer lives in the [Web UI](Web-UI); results can compare a dataset against a
  reference, which pairs well with [Perturbation](Perturbation) for building tough test sets.

Everything is scoped to your organization — see [Multi-Tenancy & Security](Multi-Tenancy-and-Security).
