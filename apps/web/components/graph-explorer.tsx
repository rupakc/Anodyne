"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphArtifact, GraphNode } from "@/lib/api";

/**
 * Self-contained, dependency-free force-directed graph explorer.
 *
 * Design constraints (spec §7 + CSP): NO external library or CDN — the layout
 * is a tiny inline force simulation and rendering is plain Canvas 2D. For large
 * graphs we never attempt to draw the whole thing: the view is reduced to a
 * **sampled subgraph** (the top-N nodes by degree plus the edges induced among
 * them), while the header always reports the true full-graph node/edge counts
 * and per-type breakdown from `metrics`.
 *
 * The animation writes node positions to a ref and paints directly to the
 * canvas, so it never triggers React re-renders. `getContext` returning null
 * (jsdom / no-canvas environments) is guarded; an always-present, visually
 * hidden node/edge summary keeps the component accessible and testable.
 */

/** Autumn-pastel hues cycled per node type; readable on light and dark canvases. */
const TYPE_COLORS = [
  "#c1694f", // terracotta
  "#d89a5d", // amber
  "#8da184", // sage
  "#c98a82", // dusty rose
  "#a8894f", // olive gold
  "#7f9b8e", // muted teal
  "#b3746a", // clay
  "#cf9f6b", // wheat
  "#9c7bb0", // muted plum
] as const;

const DEFAULT_SAMPLE_SIZE = 120;

interface Positioned {
  node: GraphNode;
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
}

function colorForType(type: string, order: string[]): string {
  const i = order.indexOf(type);
  return TYPE_COLORS[(i < 0 ? 0 : i) % TYPE_COLORS.length];
}

/** Reduce a graph to at most `sampleSize` highest-degree nodes + induced edges. */
function sampleSubgraph(graph: GraphArtifact, sampleSize: number) {
  const degree = new Map<string, number>();
  for (const e of graph.edges) {
    degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
    degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
  }
  const sampled = graph.nodes.length > sampleSize;
  const nodes = sampled
    ? [...graph.nodes]
        .sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0))
        .slice(0, sampleSize)
    : graph.nodes;
  const kept = new Set(nodes.map((n) => n.id));
  const edges = graph.edges.filter((e) => kept.has(e.source) && kept.has(e.target));
  return { nodes, edges, degree, sampled };
}

/** Full-graph aggregate stats — prefer `metrics`, fall back to counting. */
function aggregateStats(graph: GraphArtifact) {
  const m = graph.metrics ?? {};
  const nodesByType =
    m.nodes_by_type ??
    graph.nodes.reduce<Record<string, number>>((acc, n) => {
      acc[n.type] = (acc[n.type] ?? 0) + 1;
      return acc;
    }, {});
  const edgesByType =
    m.edges_by_type ??
    graph.edges.reduce<Record<string, number>>((acc, e) => {
      acc[e.type] = (acc[e.type] ?? 0) + 1;
      return acc;
    }, {});
  return {
    nodeCount: m.node_count ?? graph.nodes.length,
    edgeCount: m.edge_count ?? graph.edges.length,
    density: m.density,
    nodesByType,
    edgesByType,
  };
}

export function GraphExplorer({
  graph,
  sampleSize = DEFAULT_SAMPLE_SIZE,
  height = 420,
}: {
  graph: GraphArtifact;
  sampleSize?: number;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const posRef = useRef<Positioned[]>([]);
  const rafRef = useRef<number | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const drawRef = useRef<(() => void) | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { nodes, edges, degree, sampled } = useMemo(
    () => sampleSubgraph(graph, sampleSize),
    [graph, sampleSize],
  );
  const stats = useMemo(() => aggregateStats(graph), [graph]);

  // Ordered unique node types (ontology order first, then any extras present).
  const typeOrder = useMemo(() => {
    const order = graph.ontology?.node_types?.map((t) => t.name) ?? [];
    for (const n of nodes) if (!order.includes(n.type)) order.push(n.type);
    return order;
  }, [graph.ontology, nodes]);

  const selectedNode = selectedId ? nodes.find((n) => n.id === selectedId) ?? null : null;

  // Build + run the layout, and paint to the canvas. Positions live in a ref;
  // this effect never sets React state, so the animation is re-render free.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = containerRef.current?.clientWidth || 640;
    canvas.width = width;
    canvas.height = height;

    const cx = width / 2;
    const cy = height / 2;

    // Deterministic seeded ring layout to start (stable across renders/tests).
    posRef.current = nodes.map((node, i) => {
      const angle = (i / Math.max(1, nodes.length)) * Math.PI * 2;
      const radius = Math.min(width, height) * 0.32;
      return {
        node,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
        vx: 0,
        vy: 0,
        degree: degree.get(node.id) ?? 0,
      };
    });

    const byId = new Map(posRef.current.map((p) => [p.node.id, p]));
    const links = edges
      .map((e) => ({ s: byId.get(e.source), t: byId.get(e.target), type: e.type }))
      .filter((l): l is { s: Positioned; t: Positioned; type: string } => !!l.s && !!l.t);

    const styles = typeof getComputedStyle === "function" ? getComputedStyle(canvas) : null;
    const edgeColor = styles?.getPropertyValue("--border")?.trim() || "rgba(150,130,110,0.4)";
    const textColor = styles?.color || "#362a22";

    // getContext can be absent (SSR) or throw in jsdom (no `canvas` package);
    // guard both so the component still renders its accessible text mirror.
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext ? canvas.getContext("2d") : null;
    } catch {
      ctx = null;
    }

    function radiusFor(p: Positioned): number {
      return 5 + Math.min(9, p.degree * 1.5);
    }

    function draw() {
      if (!ctx || !canvas) return;
      const selected = selectedIdRef.current;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // edges
      ctx.lineWidth = 1;
      ctx.strokeStyle = edgeColor;
      for (const l of links) {
        const highlighted = selected && (l.s.node.id === selected || l.t.node.id === selected);
        ctx.globalAlpha = highlighted ? 0.9 : 0.35;
        ctx.beginPath();
        ctx.moveTo(l.s.x, l.s.y);
        ctx.lineTo(l.t.x, l.t.y);
        ctx.stroke();
        // Label edges by type when the graph is small, or when incident to the selection.
        if (highlighted || links.length <= 36) {
          ctx.globalAlpha = highlighted ? 0.95 : 0.6;
          ctx.fillStyle = textColor;
          ctx.font = "9px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText(l.type, (l.s.x + l.t.x) / 2, (l.s.y + l.t.y) / 2 - 2);
        }
      }
      ctx.globalAlpha = 1;

      // nodes
      for (const p of posRef.current) {
        const r = radiusFor(p);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = colorForType(p.node.type, typeOrder);
        ctx.globalAlpha = selected && p.node.id !== selected ? 0.55 : 1;
        ctx.fill();
        if (p.node.id === selected) {
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = textColor;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }
    drawRef.current = draw;

    let tick = 0;
    const maxTicks = 260;

    function step() {
      const ps = posRef.current;
      const k = 0.02;
      // Repulsion (all pairs — bounded because the subgraph is sampled).
      for (let i = 0; i < ps.length; i++) {
        for (let j = i + 1; j < ps.length; j++) {
          const a = ps[i];
          const b = ps[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 0.01) {
            dx = Math.random() - 0.5;
            dy = Math.random() - 0.5;
            d2 = 0.01;
          }
          const force = 1400 / d2;
          const d = Math.sqrt(d2);
          const fx = (dx / d) * force;
          const fy = (dy / d) * force;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
      }
      // Springs along edges.
      for (const l of links) {
        const dx = l.t.x - l.s.x;
        const dy = l.t.y - l.s.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const desired = 70;
        const f = (d - desired) * 0.02;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        l.s.vx += fx;
        l.s.vy += fy;
        l.t.vx -= fx;
        l.t.vy -= fy;
      }
      // Integrate + gentle centering + damping.
      for (const p of ps) {
        p.vx += (cx - p.x) * k * 0.08;
        p.vy += (cy - p.y) * k * 0.08;
        p.vx *= 0.82;
        p.vy *= 0.82;
        p.x += p.vx;
        p.y += p.vy;
      }
    }

    function frame() {
      step();
      draw();
      tick += 1;
      if (tick < maxTicks) {
        rafRef.current = typeof requestAnimationFrame === "function" ? requestAnimationFrame(frame) : null;
      }
    }

    draw();
    if (typeof requestAnimationFrame === "function") {
      rafRef.current = requestAnimationFrame(frame);
    }

    return () => {
      drawRef.current = null;
      if (rafRef.current != null && typeof cancelAnimationFrame === "function") {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [nodes, edges, degree, height, typeOrder]);

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / (rect.width || canvas.width);
    const scaleY = canvas.height / (rect.height || canvas.height);
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    let hit: string | null = null;
    let best = Infinity;
    for (const p of posRef.current) {
      const d2 = (p.x - mx) ** 2 + (p.y - my) ** 2;
      if (d2 < 200 && d2 < best) {
        best = d2;
        hit = p.node.id;
      }
    }
    selectedIdRef.current = hit;
    setSelectedId(hit);
    drawRef.current?.();
  }

  return (
    <div className="flex flex-col gap-4">
      <StatBar
        nodeCount={stats.nodeCount}
        edgeCount={stats.edgeCount}
        density={stats.density}
        sampled={sampled}
        sampledCount={nodes.length}
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_16rem]">
        <div
          ref={containerRef}
          className="relative overflow-hidden rounded-2xl border border-border bg-card"
        >
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            role="img"
            aria-label={`Force-directed graph: ${nodes.length} nodes and ${edges.length} edges shown${
              sampled ? ` (sampled from ${stats.nodeCount})` : ""
            }. Click a node to inspect its properties.`}
            className="w-full cursor-pointer text-foreground"
            style={{ height }}
          />
        </div>

        <aside className="flex flex-col gap-4">
          <Legend nodesByType={stats.nodesByType} typeOrder={typeOrder} />
          <SelectionPanel
            node={selectedNode}
            onClear={() => {
              selectedIdRef.current = null;
              setSelectedId(null);
              drawRef.current?.();
            }}
          />
        </aside>
      </div>

      {/* Accessible / testable text mirror of the rendered subgraph. */}
      <div className="sr-only">
        <ul aria-label="Graph nodes">
          {nodes.map((n) => (
            <li key={n.id}>
              {n.id} — {n.type}
            </li>
          ))}
        </ul>
        <ul aria-label="Graph edges">
          {edges.map((e) => (
            <li key={e.id}>
              {e.source} {e.type} {e.target}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function StatBar({
  nodeCount,
  edgeCount,
  density,
  sampled,
  sampledCount,
}: {
  nodeCount: number;
  edgeCount: number;
  density?: number;
  sampled: boolean;
  sampledCount: number;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-border bg-muted/30 px-4 py-3 text-sm">
      <Stat label="Nodes" value={nodeCount.toLocaleString()} />
      <Stat label="Edges" value={edgeCount.toLocaleString()} />
      {typeof density === "number" ? <Stat label="Density" value={density.toFixed(4)} /> : null}
      {sampled ? (
        <span className="rounded-full border border-amber/40 bg-amber/10 px-2.5 py-0.5 text-xs text-foreground">
          showing top {sampledCount.toLocaleString()} by degree
        </span>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-xs tracking-wide text-muted-foreground uppercase">{label}</span>
      <span className="font-[family-name:var(--font-data)] font-medium text-foreground">{value}</span>
    </span>
  );
}

function Legend({
  nodesByType,
  typeOrder,
}: {
  nodesByType: Record<string, number>;
  typeOrder: string[];
}) {
  const types = typeOrder.filter((t) => t in nodesByType).concat(
    Object.keys(nodesByType).filter((t) => !typeOrder.includes(t)),
  );
  if (types.length === 0) return null;
  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <h4 className="mb-2 font-[family-name:var(--font-data)] text-xs tracking-[0.2em] text-terracotta uppercase">
        Node types
      </h4>
      <ul className="flex flex-col gap-1.5" aria-label="Node type legend">
        {types.map((t) => (
          <li key={t} className="flex items-center justify-between gap-2 text-sm">
            <span className="flex items-center gap-2">
              <span
                aria-hidden
                className="size-3 rounded-full"
                style={{ backgroundColor: colorForType(t, typeOrder) }}
              />
              {t}
            </span>
            <span className="font-[family-name:var(--font-data)] text-xs text-muted-foreground">
              {nodesByType[t]?.toLocaleString()}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SelectionPanel({ node, onClear }: { node: GraphNode | null; onClear: () => void }) {
  if (!node) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card p-4 text-sm text-muted-foreground">
        Click a node to inspect its type and properties.
      </div>
    );
  }
  const entries = Object.entries(node.properties ?? {});
  return (
    <div className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-2">
        <h4 className="font-[family-name:var(--font-display)] text-sm font-semibold tracking-tight">
          {node.type}
        </h4>
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-muted-foreground underline-offset-2 hover:underline"
        >
          clear
        </button>
      </div>
      <p className="font-[family-name:var(--font-data)] text-xs break-all text-muted-foreground">{node.id}</p>
      {entries.length > 0 ? (
        <dl className="grid grid-cols-[minmax(0,auto)_1fr] gap-x-3 gap-y-1 text-xs">
          {entries.map(([k, v]) => (
            <div key={k} className="col-span-2 grid grid-cols-subgrid">
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="font-[family-name:var(--font-data)] break-all text-foreground">
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-xs text-muted-foreground">No properties.</p>
      )}
    </div>
  );
}
