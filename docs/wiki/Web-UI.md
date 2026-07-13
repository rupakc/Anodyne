# Web App

The Web App is the friendly, autumn-themed front door to everything Anodyne does. You don't need to
touch code or the command line — the whole workflow, from creating data to grading it, happens in
your browser.

## What you can do today

- **Sign in securely** with your organization's login.
- **Dashboard** — see your datasets, jobs, and their status at a glance.
- **Guided wizards for every kind of data:**
  - **Tabular** — from a plain-English description, from an uploaded sample, or from a starter
    template.
  - **Text, Image, Audio, Video** — each with its own simple wizard.
  - **Knowledge graphs** — build an ontology and generate a graph ([Graph Modality](Graph-Modality)),
    with an **interactive explorer** to visualize it and a viewer for the ontology.
- **Shape and ship your data** — panels to run [Perturbation](Perturbation) and to
  [export/download](Export-and-Storage) in the format you want.
- **Grade your data** — launch an [Evaluation](Evaluation-Engine) and read the visual report, with a
  radar chart and clickable expert cards.
- **Stay in control** — review and approve datasets and add annotations
  ([Human-in-the-Loop & Annotation](Human-in-the-Loop-and-Annotation)).
- **Bring your own AI model** — manage your AI providers and keys
  ([Bring Your Own AI Model](LLM-Abstraction)).
- **Live progress** — watch long-running jobs update in real time.

## Under the hood (in plain terms)

- It's a modern web application with a consistent, calm "autumn pastel" look that works in both
  light and dark mode.
- It talks to Anodyne's secure back end, which enforces that you only ever see **your own
  organization's** data — see [Multi-Tenancy & Security](Multi-Tenancy-and-Security).
- To try it on your own machine, see [Running Anodyne on Your Laptop](Local-Development).
