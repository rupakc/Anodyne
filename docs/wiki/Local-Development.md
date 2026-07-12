# Running Anodyne on Your Laptop

You can run the whole platform locally to try it out or develop against it. This is a high-level
overview; the exact commands live in the repository's `docs/dev-runbook.md`.

## What you need

- **Docker** (Desktop or Engine) — runs the supporting services.
- **uv** (Python) and **pnpm** (for the web app) — to run the app processes.

## The three steps

1. **Start the backbone.** One command (`make up`) launches everything the app needs: the database,
   file storage, the login server (Keycloak), the workflow engine (Temporal), the compute engine
   (Ray), and a local AI model server (Ollama) for fully offline use.
2. **Prepare the data.** `make migrate` sets up the database; `make seed` adds a demo company and user.
3. **Run the app.** `make dev` starts the API, the generation worker, and the web app together. Open
   the web app in your browser and sign in as the demo user.

## Two ways to use AI locally

- **Offline (no accounts needed).** Use the bundled Ollama model server — nothing leaves your
  machine.
- **Bring your own key.** Register a cloud model (e.g. OpenAI) with your own API key for higher
  quality.

## Trying the full flow

Once signed in: describe a table → review the suggested columns → generate → watch progress →
download the file. That exercises the entire [Generation Engine](Generation-Engine) end to end on
your laptop.

## Status

✅ Supported today. The repository's runbook has copy-paste commands and troubleshooting notes.
