# Graph Modality — Design Spec (Knowledge Graphs, Ontologies, Alignment)

**Status:** approved to implement (user delegated judgement). **Author:** design pass 2026-07-13.

Anodyne today generates tabular, text, image, audio, and video. This adds a **graph** modality:
synthetic **property graphs / knowledge graphs**, **ontology-constrained** generation, and
**ontology mapping/alignment** — slotting into the existing modality-registry + ports architecture
without touching Temporal/Ray orchestration.

## 1. Why (requirements & use cases)

1. Test data for **graph databases** (Neo4j, Neptune, TigerGraph) — realistic property graphs.
2. **GraphRAG / KG-RAG** benchmarks — synthetic KGs + multi-hop QA to test retrieval & reasoning.
3. **Ontology development** — propose an ontology (T-Box) from a domain description; generate
   conforming instance data (A-Box).
4. **Ontology mapping / alignment** — align two schemas/KGs (class/property equivalence, subsumption).
5. **GNN training data** — controllable topology (communities, homophily) + node/edge/graph labels.
6. **Privacy-safe graph release** — synthesize a graph matching a real graph's structure/attributes
   without copying subgraphs or leaking individuals.
7. **Robustness testing** — inject missing edges, noisy attributes, ontology violations.

The modality must support three sources, mirroring the rest of the platform:
**from description** (LLM-driven), **from sample** (learn + synthesize a real graph), and
**from ontology** (schema-constrained instance generation).

## 2. Canonical model (self-critiqued decision)

**Decision: the internal canonical model is a typed Property Graph (LPG)** — typed nodes and typed
edges, each with arbitrary key/value properties — because it is a superset (edges carry properties,
which RDF cannot without reification). RDF/OWL is offered as an **export projection** (edge
properties via RDF-star, falling back to reification). This avoids a full bidirectional semantic
layer in the MVP while still delivering standards-compliant RDF output.

*Rejected alternative:* RDF-triples as the canonical model. Rejected because edge properties and
LPG-native tooling (the dominant graph-DB use case) become second-class, and the reification tax
hits every operation, not just export.

Core domain models (`anodyne-graph`):
- `GraphOntology` — node types (classes), edge types (relations) with **domain/range**, datatype
  property schemas per node type, **cardinality** constraints, `subClassOf`/`subPropertyOf`
  hierarchy. This is the graph analog of C0's proposed tabular schema.
- `GraphSpec` — the generation request: ontology (optional, proposed if absent), target size
  (`node_count`, `edge_factor`/density), topology model + params, source (description/sample/
  ontology), directives (bias/edge-cases/community structure).
- `Node` (id, type, properties), `Edge` (id, type, source, target, properties), `GraphDataset`
  (nodes, edges, ontology, stats).

`target_rows` from the shared `DatasetSpec` maps to **node count**; edge count / density / topology
live in `directives`. `DatasetVersion.row_count` records `nodes + edges` with a `metrics` blob
(node/edge counts per type, density) so the UI and judges have structure at a glance.

## 3. Generation engine (the core innovation: structure + semantics)

A **hybrid** `GraphGenerator` implementing the existing `Generator` port, with pluggable strategies:

1. **Topology** — deterministic, seeded (numpy RNG per the platform idiom), via `networkx`:
   Barabási–Albert (scale-free / power-law degree), Watts–Strogatz (small-world), **Stochastic
   Block Model** and **LFR benchmark** (community structure), degree-corrected SBM, and an
   **ontology-constrained sampler** that only creates edges between domain/range-compatible node
   types respecting cardinality.
2. **Semantics** — the LLM (via the existing `LLMProvider` port) fills node attributes, relation
   labels, and realistic entity names conditioned on node type + neighborhood, in batched
   structured-output calls (reusing `anodyne-text`'s batch/dedup machinery). **PII-looking
   attributes are always faked** (Faker/Mimesis), never copied.
3. **Consistency layer** — entity canonicalization/dedup, referential integrity, and **ontology
   constraint enforcement** validated with **SHACL** shapes (a wholly-invalid graph is rejected;
   individual violations can be *intentionally* injected via directives for robustness tests).

**Determinism honesty (self-critique):** topology is bit-exact reproducible; LLM-filled semantics
are *best-effort* reproducible (temperature=0, seeded, cached) — documented, same stance as the
text modality. We do NOT over-promise bit-exact whole graphs.

**From-sample:** learn degree distribution, attribute distributions, community structure, and
edge-type mix from an uploaded graph (any supported import format), then synthesize a **new** graph
matching those statistics. Privacy via statistical matching + faked PII + a **no-verbatim-subgraph**
guarantee (checked). **Differential privacy is explicitly a future research track, NOT claimed in
MVP** (edge/node-DP is research-grade; over-claiming would be irresponsible).

## 4. Ontology mapping / alignment (advanced feature)

Given two ontologies (or a source KG + a target ontology), produce class/property **alignments**
(equivalence, subsumption) combining **lexical** (label/edit similarity), **embedding**
(semantic similarity via the LLM/embedding provider), and **LLM adjudication**, each with a
confidence score + provenance. Output in **SSSOM** (Simple Standard for Sharing Ontological
Mappings) and EDOAL. Low-confidence mappings route to **HITL review** (sub-system G) for
accept/reject — a natural tie-in.

## 5. Export & interchange (extends sub-system E `Exporter`)

- **RDF/semantic:** Turtle, N-Triples, JSON-LD, RDF/XML (via `rdflib`); ontology → **OWL**.
- **Property graph:** GraphML, GEXF, **Cypher** script, **Neo4j admin-import CSVs**, node-link JSON.
- **GNN:** PyG/DGL edge-index + feature tensors; adjacency / edge-list **Parquet** for scale.
- **Alignments:** SSSOM TSV.
Large graphs stream/chunk like the existing Parquet path (never materialize twice).

## 6. Graph-aware evaluation (extends sub-system F MoE judges)

New expert judges (each behind the `Judge` port; LLM access only via `LLMProvider`):
- **Structural fidelity** — degree-distribution KS, clustering coefficient, assortativity,
  modularity, path-length distribution, and a **scalable** graph-distance (spectral / Laplacian
  eigenvalue distance, NetLSD signatures, or Weisfeiler–Lehman graph kernel). *Self-critique: graph
  edit distance / isomorphism is NP-hard and does not scale — we deliberately use statistical +
  spectral distances instead.*
- **Ontology consistency** — SHACL validation pass-rate + OWL reasoner consistency (`pySHACL`).
- **Semantic plausibility** — LLM judge on sampled triples/subgraphs.
- **Connectivity / coverage** — giant-component fraction, isolated nodes, relation-type coverage.
- **Utility (GNN TSTR)** — train a node classifier on synthetic, test on real (lightweight, seeded).
- **Privacy / leakage** — structural re-identification risk; nearest-subgraph distance.
Aggregated into the existing weighted 360° report + JSON/HTML artifact.

## 7. UI (extends sub-system H)

Graph wizard (describe / upload sample / provide-or-propose ontology); **interactive force-directed
explorer** — *self-contained* canvas (CSP forbids external CDNs), and for large graphs a **sampled
subgraph** (top-N by centrality or a BFS neighborhood) with full-graph aggregate stats; ontology
class-hierarchy viewer; alignment review table (accept/reject → HITL); export-format picker extended
with all graph formats; graph judges surfaced in the evaluation report.

*Self-critique:* in-browser force layout of 100k+ nodes will die — hence the sampled-subgraph +
aggregate-stats approach, never a full render of an arbitrarily large graph.

## 8. Integration with the existing architecture

- `Modality.GRAPH` added to the enum; a self-registering **graph handler** in the modality registry.
- The handler fits the generation-workflow contract (`plan_shards` → `generate_shards` →
  `assemble_and_upload`): a graph is generated as **one shard** (whole graph) for typical sizes, or
  partitioned by community/node-range for very large graphs; produces a single serialized artifact.
- No changes to Temporal workflow / Ray orchestration — the payoff of the hexagonal design.
- New package `packages/anodyne-graph`; adapters live there; `anodyne-core` stays adapter-free.
- New deps (all permissive licenses): `networkx` (BSD), `rdflib` (BSD), `pySHACL` (Apache).
  Embeddings reuse the LLM provider layer. GNN utility uses the already-present sklearn (a tiny
  optional torch-geometric path is a follow-up, not MVP).

## 9. Roadmap (phased, parallelized; self-critiqued for scope)

**Scope discipline:** the full vision is 3 waves. Building everything at once would be unwieldy and
risk half-finished features (YAGNI/KISS). Phase it.

- **GA — Graph core (walking skeleton, the spine; lands first):** `Modality.GRAPH`, `anodyne-graph`
  domain models + ports, LLM-driven from-description generator, ontology model + LLM ontology
  proposer, node-link JSON export, modality handler + workflow wiring, gateway route, minimal tests.
  End-to-end one happy path — the "C0 for graphs".

- **Wave 1 (parallel worktrees, on top of GA's frozen interfaces):**
  - **GB — Generation engines:** networkx topology models, hybrid structure+semantics, from-sample
    learn+synthesize (privacy-safe, no DP), ontology-constrained generation + SHACL validation.
  - **GC — Export & interchange:** RDF (Turtle/JSON-LD/N-Triples/RDF-XML) + OWL, GraphML/GEXF/Cypher/
    Neo4j-CSV, GNN edge-index + adjacency Parquet.
  - **GD — Graph evaluation:** the six graph expert judges + aggregation into the F report.
  - **GE — Graph UI:** wizard, sampled force-directed viz, ontology viewer, export picker, report.

- **Wave 2 (advanced, later):**
  - **GF — Ontology mapping/alignment** (lexical+embedding+LLM, SSSOM, HITL review).
  - **GG — GraphRAG multi-hop QA fixtures** (questions + gold answer paths grounded in the graph).
  - **GH — Graph perturbations** (edge rewire/dropout, ontology-violation injection) extending D.
  - Research track: differential privacy for graphs.

## 10. Global constraints (bind every task)

Hexagonal (no adapter imports in `anodyne-core`); LLM only via `LLMProvider`; deterministic +
seeded (`np.random.default_rng([seed, ...])`); multi-tenant `tenant_id` + RLS on any new table;
never store/log plaintext secrets; TDD (failing test first); `mypy --strict` + `ruff` clean;
`--import-mode=importlib`, globally-unique test basenames, no `tests/__init__.py`; conventional
commits; Alembic migrations chain linearly from the current head; PII attributes always faked.
