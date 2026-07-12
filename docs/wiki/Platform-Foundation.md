# Platform Foundation

The foundation is the **secure base** every Anodyne feature is built on. You don't interact with it
directly, but it's what makes the platform safe, reliable, and usable by many companies at once.

## What it provides

- **Secure sign-in.** Anodyne uses your organization's identity system (an industry-standard called
  OIDC, via Keycloak). You log in the same way you log into other company tools.
- **Roles and permissions.** People get roles — *owner, admin, member, viewer* — that decide what
  they can do (e.g. only some roles can register AI models or delete things).
- **A place for data.** Small facts (dataset names, settings) live in a database; large files (the
  actual generated datasets) live in file storage. Both work the same whether Anodyne runs in the
  cloud or inside your own data center.
- **Health and monitoring.** Every action is logged and traceable, so operators can see what's
  happening and diagnose issues.

## Why it matters

- **Runs anywhere.** The same Anodyne can run on Google Cloud, another cloud, or fully on-premises —
  no vendor lock-in.
- **Built to scale.** It's organized as independent building blocks, so busy parts can grow without
  affecting the rest.
- **Private by design.** Each company's information is isolated from every other company's — see
  **[Multi-Tenancy & Security](Multi-Tenancy-and-Security)**.

## Status

✅ Built and in use. It's the base beneath the [Generation Engine](Generation-Engine) and everything
that follows.
