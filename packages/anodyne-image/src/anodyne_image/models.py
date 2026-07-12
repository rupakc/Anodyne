from __future__ import annotations

from pydantic import BaseModel


class GeneratedImage(BaseModel):
    """Raw output of one `ImageProvider.generate()` call."""

    data: bytes
    mime_type: str = "image/png"


class ImagePromptItem(BaseModel):
    """One deterministically-derived generation unit: which item, its (optional)
    class label, and the composed prompt text -- produced by `ImagePromptBuilder`
    before any provider is invoked.
    """

    item_index: int
    label: str | None = None
    prompt: str


class ImageManifestEntry(BaseModel):
    """One row of the final `manifest.json` written alongside the generated
    image files -- what a downstream consumer (Export, Evaluation) reads to
    know what each image is and where it lives.
    """

    item_index: int
    object_key: str
    prompt: str
    label: str | None = None
    mime_type: str = "image/png"
