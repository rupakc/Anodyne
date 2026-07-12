"""Anodyne Text: LLM-backed text corpus generation behind the `Generator` port."""

from .errors import TextGenerationError
from .prompts import build_batch_prompt
from .shapes import TextShape, detect_shape, primary_field

__all__ = [
    "TextGenerationError",
    "TextShape",
    "build_batch_prompt",
    "detect_shape",
    "primary_field",
]
