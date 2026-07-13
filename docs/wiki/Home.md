# Welcome to Anodyne

**Anodyne** helps teams create realistic **synthetic data** — made-up-but-lifelike datasets — and
then **grade how good that data is** using AI. You describe what you need in plain English, Anodyne
generates it, and you can download it or review its quality.

Think of it as a factory: you hand over a description ("a table of customers with ages and
countries"), and Anodyne builds a dataset to match — without using any real, sensitive data.

## Why people use it

- **Privacy** — test and build software without touching real customer data.
- **Speed** — get a usable dataset in minutes instead of hunting for one.
- **Control** — dial in the exact size, shape, and quirks (noise, rare cases, biases) you want.
- **Trust** — an AI "panel of judges" reviews the data and reports how realistic and useful it is.

## What you can do today

- Sign in securely (your organization's login).
- Describe a table in plain English and let Anodyne propose the columns.
- Review and tweak those columns, choose how many rows you want, and generate.
- Watch progress live, then download the finished file.

## The bigger picture

Each page below explains one piece of Anodyne in simple terms:

**Foundations**
- **[Platform Foundation](Platform-Foundation)** — the secure, multi-company base everything sits on.
- **[Bring Your Own AI Model](LLM-Abstraction)** — use any AI provider, or a private one on your own machines.
- **[Multi-Tenancy & Security](Multi-Tenancy-and-Security)** — how each company's data stays private.

**Generating data**
- **[Generation Engine](Generation-Engine)** — how tabular, text, image, audio, and video datasets get made.
- **[Graph Modality](Graph-Modality)** — generating knowledge graphs and ontologies.

**Shaping & shipping**
- **[Perturbation](Perturbation)** — deliberately roughen data to test robustness.
- **[Export & Storage](Export-and-Storage)** — download your data in the format you want.

**Judging & reviewing**
- **[Evaluation Engine](Evaluation-Engine)** — an AI panel of judges that grades the data.
- **[Human-in-the-Loop & Annotation](Human-in-the-Loop-and-Annotation)** — keep people in control, add labels and feedback.

**Using & running it**
- **[Web App](Web-UI)** — the browser interface that ties it all together.
- **[Running Anodyne on Your Laptop](Local-Development)** — trying it out locally.
- **[Deployment](Deployment)** — running Anodyne in the cloud or on-premises.
