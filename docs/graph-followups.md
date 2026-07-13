# Graph subsystem — tracked follow-ups

Known limitations surfaced during graph-modality reviews. Resolved items are
kept (struck through) for history; open items stay at the bottom.

## Resolved (Wave 2, commit 610b16a)

- **~~1. Spectral distance inflated for unequal-size graphs~~** — fixed:
  `_spectral_distance` now resamples both eigenvalue spectra to a common length
  by linear interpolation instead of zero-padding the shorter one, so a
  node-count gap no longer dominates the structural signal.
- **~~2. Empty-graph ontology score = 1.0~~** — fixed: the ontology-consistency
  judge raises `JudgeNotApplicable` for an empty graph (its weight drops out of
  the aggregate) rather than reporting a misleading perfect conformance.
- **~~3. Cypher control-character escaping~~** — fixed: `_cypher_value` escapes
  `\n`/`\r`/`\t` and any remaining C0 control char (`\uXXXX`) in string literals.
- **~~4. RDF datetime → xsd:string vs OWL range~~** — fixed: `dataset_to_rdf`
  types datetime A-Box literals as `xsd:dateTime` (looked up from the ontology's
  declared `PropertySpec.datatype`), agreeing with the OWL T-Box `rdfs:range`.

## Open

_None currently tracked._
