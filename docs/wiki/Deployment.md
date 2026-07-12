# Getting Anodyne Running for Real (Deployment & CI/CD)

Everything so far has been about running Anodyne on one laptop. This page explains, in plain
language, what it takes to run Anodyne for a real team or company — on your own servers, or on
Google Cloud.

## The idea: same boxes, everywhere

Anodyne is built from three "packages" that can each be shipped as a **container** — think of a
container like a sealed shipping crate that includes the app and everything it needs to run,
so it behaves the same on a laptop, a company's own servers, or Google Cloud:

- **The gateway** — the front door: logins, requests, and live progress updates.
- **The worker** — does the actual data-generation work in the background.
- **The web app** — what people see and click through in their browser.

Because they're sealed crates, the *same* crate can run in any of these places:

## Three ways to run it

1. **On your own servers ("on-prem").** One command starts everything — the database, the login
   system, and the three crates above — using Docker Compose, a tool for running several
   containers together as one system.
2. **Google Cloud Run.** A "serverless" option: Google runs the containers for you and
   automatically adds more copies when traffic increases, without anyone managing servers by hand.
3. **Google Kubernetes Engine (GKE).** The full, most powerful setup — used when a company needs
   the heavier compute engine (for generating images, audio, or video) or wants everything
   running at larger scale.

All three ultimately run the exact same containers — nothing behaves differently depending on
where it's deployed.

## Building and checking the containers automatically

Every time a change is proposed to Anodyne, an automated pipeline:

1. **Builds** the three containers fresh.
2. **Lists every ingredient** that went into them (a "software bill of materials," or SBOM) — like
   a nutrition label for software, so anyone can check exactly what's inside later.
3. **Scans for known security problems** and stops the process if anything serious turns up.
4. Only if all of that passes, **ships** the containers to Google Cloud, ready to deploy.

## How secrets are kept safe

Passwords, API keys, and other secrets are never written into the code or the container recipes.
Instead:
- On a laptop or company server, they live in a local settings file that's never shared or checked
  into the project's history.
- In Google Cloud, they live in Google's own secret-storage vault, and the containers are only
  given permission to read the specific secret they need, at the moment they start up.
- The automated pipeline that ships containers to Google Cloud doesn't use a stored password at
  all — it proves who it is on the fly, each time, using a short-lived, single-use credential that
  expires almost immediately. Nothing long-lived is ever stored that a leak could expose.

## Status

🚧 **In progress → scaffolded.** The container recipes, the automated build/check/ship pipeline,
and the deployment blueprints for on-prem, Cloud Run, and GKE all exist and have been checked for
correctness. Turning them on against a real Google Cloud account (so containers actually start
running there) is the next, separate step — it needs a company to provide real cloud credentials
first, which this build didn't have. See the repository's `docs/deployment.md` for the technical
detail.
