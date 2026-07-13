# Graph subsystem — tracked follow-ups

Known limitations surfaced during the Wave-1 integration review. These are
**intentionally deferred** (not bugs blocking the merge); each is low-severity
or an accuracy refinement. Filed here so they are not lost.

## 1. Spectral distance inflated for unequal-size graphs
The spectral graph-distance metric pads the shorter eigenvalue spectrum with
zeros before comparing. For two graphs of different node counts this padding
inflates the reported distance (the size gap dominates the structural signal).
Follow-up: normalize/resample spectra to a common length, or switch to a
size-invariant spectral descriptor, before scoring cross-size comparisons.

## 2. Empty-graph ontology score = 1.0
The ontology-conformance score returns a perfect `1.0` for an empty graph
(no nodes/edges to violate any constraint). This is vacuously true but
misleading in reports. Follow-up: treat an empty graph as unscored (`None`) or
a defined sentinel rather than a perfect conformance.

## 3. Cypher control-character escaping
`dataset_to_cypher` escapes backslashes and double quotes in string values but
not control characters (newlines, tabs, `\r`, other C0 chars). A property value
containing a raw newline produces a Cypher string literal that some drivers
reject. Follow-up: escape control characters (`\n`, `\t`, `\r`, `\uXXXX`) when
emitting Cypher string literals.

## 4. RDF datetime → xsd:string vs OWL range
Instance-data (`dataset_to_rdf`) emits datetime property values as plain
string literals, while the OWL T-Box (`ontology_to_owl`) declares those same
properties with `rdfs:range xsd:dateTime`. The A-Box literal datatype therefore
disagrees with the declared T-Box range. Follow-up: type datetime literals as
`xsd:dateTime` (and reconcile other datatypes) so instance data validates
against the emitted ontology.
