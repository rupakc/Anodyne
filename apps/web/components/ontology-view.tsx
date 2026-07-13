"use client";

import { ArrowRight } from "lucide-react";
import type { GraphOntology, GraphNodeType, GraphEdgeType } from "@/lib/api";
import { EmptyState } from "@/components/ui/feedback";

/**
 * Read-only viewer for a graph ontology (T-Box): node types with their
 * datatype properties, edge types rendered as `source → target`, and any
 * `subClassOf` hierarchy. Self-contained and theme-aware (autumn-pastel CSS
 * vars); no external deps. Used both inside the graph wizard's review step
 * and on the dataset detail page.
 */
export function OntologyView({ ontology }: { ontology?: GraphOntology | null }) {
  const nodeTypes = ontology?.node_types ?? [];
  const edgeTypes = ontology?.edge_types ?? [];

  if (nodeTypes.length === 0 && edgeTypes.length === 0) {
    return <EmptyState>No ontology proposed yet.</EmptyState>;
  }

  const hasHierarchy = nodeTypes.some((n) => n.subclass_of);

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h3 className="mb-3 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Node types ({nodeTypes.length})
        </h3>
        <ul className="grid gap-3 sm:grid-cols-2" aria-label="Node types">
          {nodeTypes.map((nt) => (
            <NodeTypeCard key={nt.name} nodeType={nt} />
          ))}
        </ul>
      </section>

      <section>
        <h3 className="mb-3 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
          Edge types ({edgeTypes.length})
        </h3>
        {edgeTypes.length === 0 ? (
          <p className="text-sm text-muted-foreground">No edge types defined.</p>
        ) : (
          <ul className="flex flex-col gap-2" aria-label="Edge types">
            {edgeTypes.map((et) => (
              <EdgeTypeRow key={`${et.name}-${et.source_type}-${et.target_type}`} edgeType={et} />
            ))}
          </ul>
        )}
      </section>

      {hasHierarchy ? (
        <section>
          <h3 className="mb-3 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
            Subclass hierarchy
          </h3>
          <ul className="flex flex-col gap-1 text-sm" aria-label="Subclass hierarchy">
            {nodeTypes
              .filter((n) => n.subclass_of)
              .map((n) => (
                <li key={n.name} className="flex items-center gap-2 text-muted-foreground">
                  <span className="font-[family-name:var(--font-data)] text-foreground">{n.name}</span>
                  <ArrowRight className="size-3.5 text-dusty-rose" aria-label="subclass of" />
                  <span className="font-[family-name:var(--font-data)] text-foreground">{n.subclass_of}</span>
                </li>
              ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function NodeTypeCard({ nodeType }: { nodeType: GraphNodeType }) {
  return (
    <li className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-4 shadow-sm">
      <p className="flex items-center gap-2 font-[family-name:var(--font-display)] text-base font-semibold tracking-tight">
        <span className="size-2.5 rounded-full bg-terracotta" aria-hidden />
        {nodeType.name}
      </p>
      {nodeType.properties.length > 0 ? (
        <ul className="flex flex-wrap gap-1.5" aria-label={`${nodeType.name} properties`}>
          {nodeType.properties.map((p) => (
            <li
              key={p.name}
              className="rounded-full border border-border bg-background px-2.5 py-0.5 font-[family-name:var(--font-data)] text-xs text-muted-foreground"
            >
              {p.name} <span className="text-amber">· {p.datatype}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-muted-foreground">No properties.</p>
      )}
    </li>
  );
}

function EdgeTypeRow({ edgeType }: { edgeType: GraphEdgeType }) {
  return (
    <li className="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card px-4 py-2.5 text-sm shadow-sm">
      <span className="rounded-md bg-sage/15 px-2 py-0.5 font-[family-name:var(--font-data)] text-xs text-foreground">
        {edgeType.source_type}
      </span>
      <span className="flex items-center gap-1.5 text-dusty-rose">
        <ArrowRight className="size-4" aria-hidden />
        <span className="font-[family-name:var(--font-data)] text-xs font-medium tracking-wide text-foreground uppercase">
          {edgeType.name}
        </span>
        <ArrowRight className="size-4" aria-hidden />
      </span>
      <span className="rounded-md bg-amber/15 px-2 py-0.5 font-[family-name:var(--font-data)] text-xs text-foreground">
        {edgeType.target_type}
      </span>
    </li>
  );
}
