"""Ports for ontology mapping (adapter-free).

`EntityMatcher` is the protocol the `OntologyAligner` depends on so it is
decoupled from the concrete lexical/embedding matchers. `EmbeddingFn` is the
injected embedding function the `EmbeddingMatcher` uses in place of a
(nonexistent) `LLMProvider.embed` method — tests pass a deterministic fake.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

# Maps a batch of texts to their embedding vectors (row-aligned). Injected so
# the platform can back it with any embedding provider without the domain
# taking an adapter dependency; tests inject a deterministic fake.
EmbeddingFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@runtime_checkable
class EntityMatcher(Protocol):
    """Scores a candidate `(source, target)` entity-label pair in [0, 1].

    `name` is provenance recorded on the produced `Mapping.matcher`. `available`
    lets a matcher (e.g. embedding without an injected function) declare itself
    inactive so the aligner skips it gracefully.
    """

    name: str
    available: bool

    def score(self, source: str, target: str) -> float: ...
