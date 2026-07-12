# Bring Your Own AI Model

Anodyne uses AI models (the technology behind tools like ChatGPT) to understand your descriptions
and help build datasets. A core principle: **you choose which AI it uses.**

## What this means for you

- **Any provider.** Use OpenAI, Anthropic, Google, and 100+ others — whatever your organization
  prefers or already pays for.
- **Private / offline models too.** Prefer not to send data to an outside service? Run a model on
  your own hardware (via tools like Ollama or vLLM). Anodyne treats it the same as any cloud model.
- **Per-company keys, kept secret.** Each company registers its own AI models and access keys. Those
  keys are **encrypted** before they're stored and are **never shown back** in the interface or logs.

## Why it's designed this way

- **No lock-in.** If a better or cheaper model comes along, switch to it without changing how
  Anodyne works.
- **Cost and compliance control.** You use accounts and models that fit your budget and data-handling
  rules.
- **Offline capability.** Sensitive environments can run fully self-contained.

## How it works, simply

Anodyne has one internal "adapter" that speaks to all these models through a single, consistent
interface. Add a model once, and every Anodyne feature (generation today; evaluation later) can use
it.

## Status

✅ Built. Models are registered per company and used by the [Generation Engine](Generation-Engine).
