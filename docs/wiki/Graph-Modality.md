# Graph Modality

Not all data fits in a table. **Knowledge graphs** capture *things* and the *relationships between
them* — people who work at companies, products in categories, concepts linked to other concepts.
Anodyne can generate these graphs (and their "ontologies," the rulebook of allowed types) just as
easily as it generates tables.

## What you can do today

- **Define the shape (ontology), or let AI propose one.** An ontology lists the kinds of things
  (node types), the kinds of connections (edge types), and the properties each can have. Describe
  what you want in plain English and Anodyne can draft the ontology for you.
- **Generate a realistic graph.** Anodyne combines two ingredients: a realistic **network
  structure** (how things connect — some hubs, many smaller nodes, communities) and **AI-filled
  details** that obey your ontology's rules. You can also **learn from a sample** graph, safely —
  personal-looking details are always faked, never copied.
- **Export to the format your tools use:**
  - **Semantic web:** RDF, OWL, Turtle, JSON-LD.
  - **Property graph:** GraphML, GEXF, Cypher, Neo4j CSV.
  - **Graph machine learning:** GNN-ready formats.
- **Explore it visually** in the browser, and browse the ontology — all in the [Web App](Web-UI).

### Going further

- **Ontology mapping & alignment** — match one ontology's terms to another's, exported as a standard
  **SSSOM** mapping, with a human review step for uncertain matches
  ([Human-in-the-Loop & Annotation](Human-in-the-Loop-and-Annotation)).
- **GraphRAG question sets** — auto-generate multi-hop question-and-answer fixtures whose answers are
  **grounded in the graph itself** (never made up), for testing retrieval and reasoning systems.
- **Graph-specific grading** — the [Evaluation Engine](Evaluation-Engine) judges graphs on
  structure, ontology consistency, connectivity, privacy, learning-usefulness, and plausibility.
- **Graph perturbations** — rewire connections, drop nodes/edges, or inject rule-breaking records to
  build hard test cases ([Perturbation](Perturbation)).

## Under the hood (in plain terms)

- Graphs use the same reliable generation, [export](Export-and-Storage), and evaluation machinery as
  every other modality — nothing new to learn operationally.
- The AI-filled parts go through your configured model ([Bring Your Own AI Model](LLM-Abstraction)),
  and everything stays private to your organization
  ([Multi-Tenancy & Security](Multi-Tenancy-and-Security)).
