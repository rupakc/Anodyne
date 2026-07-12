from __future__ import annotations

from collections.abc import Mapping


def passes_quality(
    row: Mapping[str, object],
    primary: str,
    min_length: int = 1,
    max_length: int = 20_000,
) -> bool:
    """Basic quality gate on a generated row's primary text field.

    Rejects a row whose primary field is missing, not a string, empty/
    whitespace-only after stripping, or outside `[min_length, max_length]`
    (measured on the stripped value).
    """
    value = row.get(primary)
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return min_length <= len(stripped) <= max_length


class Deduplicator:
    """Exact-match (post-strip) deduplication on a row's primary field value.

    Stateful and per-instance: construct a fresh one per shard/generation run
    -- there is no cross-shard/cross-job dedup in C2 (see design doc non-goals).
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, row: Mapping[str, object], primary: str) -> bool:
        value = row.get(primary)
        key = value.strip() if isinstance(value, str) else repr(value)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False
