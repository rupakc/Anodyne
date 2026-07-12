"""Anodyne Text: LLM-backed text corpus generation behind the `Generator` port."""

from .errors import TextGenerationError
from .generator import TextGenerator
from .prompts import build_batch_prompt
from .quality import Deduplicator, passes_quality
from .shapes import TextShape, detect_shape, primary_field

__all__ = [
    "Deduplicator",
    "TextGenerationError",
    "TextGenerator",
    "TextShape",
    "build_batch_prompt",
    "detect_shape",
    "passes_quality",
    "primary_field",
]
