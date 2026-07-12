from __future__ import annotations


class TextGenerationError(Exception):
    """Raised when text generation cannot produce any valid rows for a shard."""
