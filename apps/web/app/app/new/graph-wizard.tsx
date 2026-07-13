"use client";

import { useMemo, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import {
  createApiClient,
  GRAPH_TOPOLOGIES,
  type ApiClient,
  type DatasetSpec,
  type GraphOntology,
  type GraphSource,
  type GraphTopology,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Field, TextInput, TextArea, Select } from "@/components/ui/form";
import { ErrorAlert, InfoNote } from "@/components/ui/feedback";
import { ProviderSelect } from "@/components/provider-select";
import { OntologyView } from "@/components/ontology-view";
import { StepIndicator, type WizardStepMeta } from "./step-indicator";

type Step = "describe" | "ontology" | "confirm";

const STEPS: WizardStepMeta[] = [
  { key: "describe", label: "Domain" },
  { key: "ontology", label: "Review ontology" },
  { key: "confirm", label: "Generate" },
];

const SOURCES: { key: GraphSource; label: string; hint: string }[] = [
  { key: "description", label: "From a description", hint: "The LLM proposes an ontology from your domain text." },
  { key: "sample", label: "From a sample graph", hint: "Upload a GraphML/JSON graph to learn its shape (preview)." },
  { key: "ontology", label: "From an ontology", hint: "Provide your own ontology; instance data conforms to it." },
];

/** Pull the ontology out of a dataset's directives, tolerating shape drift. */
function ontologyOf(spec: DatasetSpec | null): GraphOntology | null {
  const raw = (spec?.directives as { ontology?: GraphOntology } | undefined)?.ontology;
  return raw && Array.isArray(raw.node_types) ? raw : null;
}

export interface GraphWizardProps {
  accessToken?: string;
  api?: ApiClient;
}

/**
 * Knowledge-graph generation flow: describe a domain (or upload a sample /
 * provide an ontology) → review-and-edit the proposed ontology → set node
 * count + topology → pick an LLM model + review gate → generate. Creates the
 * dataset with `modality:"graph"`; the gateway returns a proposed ontology in
 * `directives.ontology`. From-sample/from-ontology fields are wired now and
 * degrade gracefully where the backend route (GB) has not yet landed.
 */
export function GraphWizard({ accessToken, api: injectedApi }: GraphWizardProps) {
  const router = useRouter();
  const api = useMemo(() => injectedApi ?? createApiClient(accessToken), [injectedApi, accessToken]);

  const [step, setStep] = useState<Step>("describe");
  const [name, setName] = useState("");
  const [source, setSource] = useState<GraphSource>("description");
  const [description, setDescription] = useState("");
  const [nodeCount, setNodeCount] = useState(500);
  const [topology, setTopology] = useState<GraphTopology>("ontology_constrained");
  const [sampleFile, setSampleFile] = useState<File | null>(null);
  const [ontologyText, setOntologyText] = useState("");

  const [spec, setSpec] = useState<DatasetSpec | null>(null);
  const [ontologyDraft, setOntologyDraft] = useState("");
  const [editing, setEditing] = useState(false);
  const [ontologyErr, setOntologyErr] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [requireReview, setRequireReview] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit =
    name.trim().length > 0 &&
    nodeCount > 0 &&
    (source !== "ontology" || ontologyText.trim().length > 0);

  async function handleDescribe(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setError(null);

    const directives: Record<string, unknown> = {
      topology: { model: topology },
    };
    if (source === "ontology") {
      try {
        directives.ontology = JSON.parse(ontologyText);
      } catch {
        setError("The ontology must be valid JSON (node_types / edge_types).");
        setPending(false);
        return;
      }
    }
    if (source === "sample" && sampleFile) {
      directives.sample_filename = sampleFile.name;
    }

    try {
      const created = await api.createGraphDataset({
        name: name.trim(),
        description: description.trim(),
        target_rows: nodeCount,
        source,
        directives,
      });

      // From-sample: attach the uploaded graph best-effort (route may be GB-pending).
      if (source === "sample" && sampleFile) {
        try {
          await api.uploadSample(created.id, sampleFile);
        } catch {
          // Tolerated — generation still proceeds from the proposed ontology.
        }
      }

      setSpec(created);
      const onto = ontologyOf(created);
      setOntologyDraft(onto ? JSON.stringify(onto, null, 2) : "");
      setStep("ontology");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to propose an ontology. Please try again.");
    } finally {
      setPending(false);
    }
  }

  async function handleOntologyNext() {
    if (!spec) return;
    setPending(true);
    setError(null);
    setOntologyErr(null);

    // If the reviewer edited the ontology JSON, persist it via PATCH.
    if (editing && ontologyDraft.trim()) {
      let parsed: GraphOntology;
      try {
        parsed = JSON.parse(ontologyDraft);
      } catch {
        setOntologyErr("Ontology JSON is invalid — fix it or cancel editing.");
        setPending(false);
        return;
      }
      try {
        const updated = await api.updateDataset(spec.id, {
          directives: { ...spec.directives, ontology: parsed },
        });
        setSpec(updated);
        setEditing(false);
      } catch (err) {
        // PATCH may not be merged yet — tolerate and continue with the draft.
        setSpec({ ...spec, directives: { ...spec.directives, ontology: parsed } });
        setEditing(false);
        setOntologyErr(
          err instanceof Error ? `Could not save edits to the server (${err.message}); continuing locally.` : null,
        );
      }
    }
    setPending(false);
    setStep("confirm");
  }

  async function handleGenerate() {
    if (!spec) return;
    setPending(true);
    setError(null);
    try {
      const job = await api.generate(spec.id, {
        model_config_id: modelId || undefined,
        require_review: requireReview,
      });
      router.push(`/app/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start generation. Please try again.");
      setPending(false);
    }
  }

  const ontology = ontologyOf(spec);

  return (
    <div className="flex flex-col gap-8">
      <StepIndicator steps={STEPS} current={step} />

      {error ? <ErrorAlert>{error}</ErrorAlert> : null}

      {step === "describe" ? (
        <form
          onSubmit={handleDescribe}
          className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm"
        >
          <div>
            <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Step 1 of 3
            </p>
            <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
              Describe your knowledge graph
            </h2>
            <p className="mt-2 text-sm text-muted-foreground text-pretty">
              Anodyne proposes an ontology (node &amp; edge types) you review next, then generates a
              typed property graph.
            </p>
          </div>

          <Field label="Dataset name" htmlFor="graph-name">
            <TextInput
              id="graph-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Biomedical knowledge graph"
              required
            />
          </Field>

          <fieldset className="flex flex-col gap-2">
            <legend className="mb-1 text-sm font-medium">Source</legend>
            <div className="grid gap-2 sm:grid-cols-3">
              {SOURCES.map((s) => (
                <label
                  key={s.key}
                  className={
                    "flex cursor-pointer flex-col gap-1 rounded-xl border p-3 text-sm transition-colors " +
                    (source === s.key
                      ? "border-terracotta bg-terracotta/10"
                      : "border-border bg-background hover:bg-muted")
                  }
                >
                  <span className="flex items-center gap-2 font-medium">
                    <input
                      type="radio"
                      name="graph-source"
                      value={s.key}
                      checked={source === s.key}
                      onChange={() => setSource(s.key)}
                      className="accent-terracotta"
                    />
                    {s.label}
                  </span>
                  <span className="pl-6 text-xs text-muted-foreground text-pretty">{s.hint}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <Field
            label={source === "ontology" ? "Domain guidance" : "Domain description"}
            htmlFor="graph-description"
            hint={
              source === "description"
                ? "Describe the entities and how they relate."
                : "Optional — extra guidance for the generator."
            }
          >
            <TextArea
              id="graph-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Genes, proteins, and diseases with regulates / associated_with relations."
            />
          </Field>

          {source === "sample" ? (
            <Field label="Sample graph file" htmlFor="graph-sample" hint="GraphML or node-link JSON. Preview — from-sample may be pending.">
              <input
                id="graph-sample"
                type="file"
                accept=".graphml,.xml,.json"
                onChange={(e) => setSampleFile(e.target.files?.[0] ?? null)}
                className="text-sm file:mr-3 file:rounded-lg file:border file:border-border file:bg-muted file:px-3 file:py-1.5 file:text-sm"
              />
            </Field>
          ) : null}

          {source === "ontology" ? (
            <Field
              label="Ontology (JSON)"
              htmlFor="graph-ontology-input"
              hint='e.g. {"node_types":[{"name":"Gene","properties":[{"name":"symbol","datatype":"string"}]}],"edge_types":[{"name":"regulates","source_type":"Gene","target_type":"Gene"}]}'
            >
              <TextArea
                id="graph-ontology-input"
                value={ontologyText}
                onChange={(e) => setOntologyText(e.target.value)}
                className="font-[family-name:var(--font-data)] text-xs"
                placeholder='{"node_types": [...], "edge_types": [...]}'
              />
            </Field>
          ) : null}

          <div className="grid gap-6 sm:grid-cols-2">
            <Field label="Node count" htmlFor="graph-node-count">
              <TextInput
                id="graph-node-count"
                type="number"
                min={1}
                value={nodeCount}
                onChange={(e) => setNodeCount(Number(e.target.value) || 0)}
              />
            </Field>
            <Field label="Topology" htmlFor="graph-topology" hint={GRAPH_TOPOLOGIES.find((t) => t.value === topology)?.hint}>
              <Select
                id="graph-topology"
                value={topology}
                onChange={(e) => setTopology(e.target.value as GraphTopology)}
              >
                {GRAPH_TOPOLOGIES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>

          <div className="flex justify-end">
            <Button type="submit" size="lg" disabled={!canSubmit || pending}>
              {pending ? "Proposing ontology…" : "Propose ontology"}
            </Button>
          </div>
        </form>
      ) : null}

      {step === "ontology" && spec ? (
        <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
          <div>
            <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Step 2 of 3
            </p>
            <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
              Review the proposed ontology
            </h2>
            <p className="mt-2 text-sm text-muted-foreground text-pretty">
              &ldquo;{spec.name}&rdquo; — {ontology?.node_types.length ?? 0} node types,{" "}
              {ontology?.edge_types.length ?? 0} edge types. Edit before generating if needed.
            </p>
          </div>

          {ontologyErr ? <InfoNote>{ontologyErr}</InfoNote> : null}

          {editing ? (
            <Field label="Ontology (JSON)" htmlFor="graph-ontology-edit">
              <TextArea
                id="graph-ontology-edit"
                value={ontologyDraft}
                onChange={(e) => setOntologyDraft(e.target.value)}
                className="min-h-64 font-[family-name:var(--font-data)] text-xs"
              />
            </Field>
          ) : (
            <OntologyView ontology={ontology} />
          )}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setOntologyErr(null);
                if (!editing && !ontologyDraft) {
                  setOntologyDraft(ontology ? JSON.stringify(ontology, null, 2) : "");
                }
                setEditing((v) => !v);
              }}
            >
              {editing ? "Done editing" : "Edit as JSON"}
            </Button>
            <div className="flex gap-3">
              <Button type="button" variant="outline" onClick={() => setStep("describe")} disabled={pending}>
                Back
              </Button>
              <Button type="button" size="lg" onClick={handleOntologyNext} disabled={pending}>
                {pending ? "Saving…" : "Save & continue"}
              </Button>
            </div>
          </div>
        </section>
      ) : null}

      {step === "confirm" && spec ? (
        <section className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-8 shadow-sm">
          <div>
            <p className="font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
              Step 3 of 3
            </p>
            <h2 className="mt-2 font-[family-name:var(--font-display)] text-2xl font-semibold tracking-tight text-balance">
              Confirm &amp; generate
            </h2>
            <p className="mt-2 text-sm text-muted-foreground text-pretty">
              Pick the LLM model that fills node attributes and relation labels, then generate.
            </p>
          </div>

          <dl className="grid grid-cols-2 gap-4 rounded-xl border border-border bg-muted/40 p-5 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs tracking-wide text-muted-foreground uppercase">Node types</dt>
              <dd className="mt-1 font-medium">{ontology?.node_types.length ?? 0}</dd>
            </div>
            <div>
              <dt className="text-xs tracking-wide text-muted-foreground uppercase">Edge types</dt>
              <dd className="mt-1 font-medium">{ontology?.edge_types.length ?? 0}</dd>
            </div>
            <div>
              <dt className="text-xs tracking-wide text-muted-foreground uppercase">Nodes</dt>
              <dd className="mt-1 font-medium">{nodeCount.toLocaleString()}</dd>
            </div>
            <div>
              <dt className="text-xs tracking-wide text-muted-foreground uppercase">Topology</dt>
              <dd className="mt-1 font-medium">{GRAPH_TOPOLOGIES.find((t) => t.value === topology)?.label}</dd>
            </div>
          </dl>

          <ProviderSelect
            api={api}
            kind="models"
            label="LLM model (for node/edge semantics)"
            value={modelId}
            onChange={setModelId}
            allowAuto
            id="graph-model"
          />

          <label className="flex items-start gap-3 rounded-xl border border-border bg-muted/40 p-4 text-sm">
            <input
              type="checkbox"
              className="mt-0.5 size-4 accent-terracotta"
              checked={requireReview}
              disabled={pending}
              onChange={(e) => setRequireReview(e.target.checked)}
            />
            <span>
              <span className="font-medium">Require human review before generating</span>
              <span className="mt-1 block text-muted-foreground text-pretty">
                Park this job for a reviewer to approve before any graph is generated. Off by default.
              </span>
            </span>
          </label>

          <div className="flex justify-between">
            <Button type="button" variant="outline" onClick={() => setStep("ontology")} disabled={pending}>
              Back
            </Button>
            <Button type="button" size="lg" onClick={handleGenerate} disabled={pending}>
              {pending ? "Starting generation…" : "Generate graph"}
            </Button>
          </div>
        </section>
      ) : null}
    </div>
  );
}
