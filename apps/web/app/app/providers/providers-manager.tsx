"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Trash2 } from "lucide-react";
import {
  createApiClient,
  HTTPS_REQUIRED_MESSAGE,
  isCredentialSubmissionInsecure,
  PROVIDER_KINDS,
  type ApiClient,
  type ProviderConfig,
  type ProviderKind,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Field, TextArea, TextInput } from "@/components/ui/form";
import { EmptyState, ErrorAlert, Loading, SectionHeading } from "@/components/ui/feedback";
import { cn } from "@/lib/utils";

export interface ProvidersManagerProps {
  /** OIDC access token from `session.accessToken`. */
  accessToken?: string;
  /** Dependency-injection point: pass a fake/mock `ApiClient` in tests. */
  api?: ApiClient;
}

type LoadState = "loading" | "ready" | "error";

/**
 * Manage the tenant's per-modality provider registries (`/models`,
 * `/image-providers`, `/audio-providers`, `/video-providers`). A tab per
 * registry lists the registered providers (secret_ref is stripped
 * server-side, so nothing sensitive is ever shown) and offers a register
 * form + per-row delete.
 */
export function ProvidersManager({ accessToken, api: injectedApi }: ProvidersManagerProps) {
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);
  const [kind, setKind] = useState<ProviderKind>(PROVIDER_KINDS[0].kind);
  const [providers, setProviders] = useState<ProviderConfig[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const activeMeta = PROVIDER_KINDS.find((k) => k.kind === kind)!;

  useEffect(() => {
    let cancelled = false;
    api.listProviders(kind).then(
      (list) => {
        if (cancelled) return;
        setProviders(list);
        setError(null);
        setState("ready");
      },
      (err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load providers.");
        setState("error");
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, kind]);

  function selectKind(next: ProviderKind) {
    if (next === kind) return;
    setKind(next);
    setState("loading");
    setError(null);
  }

  async function refresh() {
    try {
      const list = await api.listProviders(kind);
      setProviders(list);
    } catch {
      // Leave the current list in place on a transient refresh failure.
    }
  }

  async function handleDelete(id: string) {
    setDeletingId(id);
    setError(null);
    try {
      await api.deleteProvider(kind, id);
      setProviders((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete the provider.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <SectionHeading
        eyebrow="Providers"
        title="Provider registry"
        description="Register the model/GPU providers each modality generates with. Secrets are stored server-side and never returned here."
      />

      <div role="tablist" aria-label="Provider modality" className="flex flex-wrap gap-2">
        {PROVIDER_KINDS.map((k) => {
          const active = k.kind === kind;
          return (
            <button
              key={k.kind}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => selectKind(k.kind)}
              className={cn(
                "rounded-lg border px-3.5 py-1.5 text-sm font-medium transition-colors",
                active
                  ? "border-terracotta bg-terracotta text-terracotta-foreground"
                  : "border-border bg-background text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              {k.label}
            </button>
          );
        })}
      </div>

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      <section aria-label={`Registered ${activeMeta.label} providers`}>
        {state === "loading" ? (
          <Loading label={`Loading ${activeMeta.label} providers…`} />
        ) : providers.length === 0 ? (
          <EmptyState>
            No {activeMeta.label} providers registered yet. Add one below to enable{" "}
            {activeMeta.modality === "llm" ? "text/LLM" : activeMeta.modality} generation.
          </EmptyState>
        ) : (
          <ul className="flex flex-col gap-3" aria-label="Providers">
            {providers.map((p) => (
              <li
                key={p.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 shadow-sm"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="font-medium">{p.name}</p>
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 font-[family-name:var(--font-data)] text-[0.65rem] tracking-wide uppercase",
                        p.enabled
                          ? "border-sage/50 bg-sage/20 text-ink"
                          : "border-border bg-muted text-muted-foreground",
                      )}
                    >
                      {p.enabled ? "enabled" : "disabled"}
                    </span>
                  </div>
                  <p className="mt-1 font-[family-name:var(--font-data)] text-sm text-muted-foreground">
                    {p.provider}/{p.model}
                  </p>
                  {p.api_base ? (
                    <p className="mt-0.5 truncate font-[family-name:var(--font-data)] text-xs text-muted-foreground">
                      {p.api_base}
                    </p>
                  ) : null}
                </div>
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDelete(p.id)}
                  disabled={deletingId === p.id}
                  aria-label={`Delete ${p.name}`}
                >
                  <Trash2 className="size-3.5" data-icon="inline-start" />
                  {deletingId === p.id ? "Removing…" : "Delete"}
                </Button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <RegisterProviderForm
        key={kind}
        modalityLabel={activeMeta.label}
        onSubmit={async (input) => {
          await api.registerProvider(kind, input);
          await refresh();
        }}
      />
    </div>
  );
}

function RegisterProviderForm({
  modalityLabel,
  onSubmit,
}: {
  modalityLabel: string;
  onSubmit: (input: {
    name: string;
    provider: string;
    model: string;
    api_key?: string;
    api_base?: string;
    params?: Record<string, unknown>;
  }) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [apiBase, setApiBase] = useState("");
  const [paramsText, setParamsText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const canSubmit = name.trim() && provider.trim() && model.trim();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    // Never let a secret leave the browser in cleartext: block submitting an
    // api_key when the gateway is plain http:// to a non-loopback host.
    if (apiKey.trim() && isCredentialSubmissionInsecure()) {
      setError(HTTPS_REQUIRED_MESSAGE);
      return;
    }

    let params: Record<string, unknown> | undefined;
    if (paramsText.trim()) {
      try {
        params = JSON.parse(paramsText) as Record<string, unknown>;
      } catch {
        setError("Params must be valid JSON (or left blank).");
        return;
      }
    }

    setBusy(true);
    setError(null);
    setDone(false);
    try {
      await onSubmit({
        name: name.trim(),
        provider: provider.trim(),
        model: model.trim(),
        api_key: apiKey.trim() || undefined,
        api_base: apiBase.trim() || undefined,
        params,
      });
      setName("");
      setProvider("");
      setModel("");
      setApiKey("");
      setApiBase("");
      setParamsText("");
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to register the provider.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
    >
      <div>
        <h2 className="font-[family-name:var(--font-display)] text-lg font-semibold tracking-tight">
          Register a {modalityLabel} provider
        </h2>
        <p className="mt-1 text-sm text-muted-foreground text-pretty">
          The API key is sent once and stored as a secret reference — it is never
          displayed again.
        </p>
      </div>

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}
      {done ? (
        <p className="rounded-xl border border-sage/40 bg-sage/10 px-4 py-3 text-sm text-foreground">
          Provider registered.
        </p>
      ) : null}

      <div className="grid gap-6 sm:grid-cols-2">
        <Field label="Name" htmlFor="prov-name">
          <TextInput
            id="prov-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Local Ollama"
            required
          />
        </Field>
        <Field label="Provider" htmlFor="prov-provider" hint="e.g. ollama, openai, replicate, stability">
          <TextInput
            id="prov-provider"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            placeholder="ollama"
            required
          />
        </Field>
        <Field label="Model" htmlFor="prov-model">
          <TextInput
            id="prov-model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="llama3.1 / sdxl / whisper-1"
            required
          />
        </Field>
        <Field label="API key" htmlFor="prov-key" hint="Optional for local providers.">
          <TextInput
            id="prov-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="••••••••"
            autoComplete="off"
          />
        </Field>
        <Field label="API base" htmlFor="prov-base" hint="Optional custom endpoint.">
          <TextInput
            id="prov-base"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://localhost:11434"
          />
        </Field>
        <Field label="Params (JSON)" htmlFor="prov-params" hint="Optional extra config, e.g. {&quot;temperature&quot;: 0.2}">
          <TextArea
            id="prov-params"
            value={paramsText}
            onChange={(e) => setParamsText(e.target.value)}
            placeholder="{}"
            className="font-[family-name:var(--font-data)] text-xs"
          />
        </Field>
      </div>

      <div className="flex justify-end">
        <Button type="submit" size="lg" disabled={!canSubmit || busy}>
          {busy ? "Registering…" : "Register provider"}
        </Button>
      </div>
    </form>
  );
}
