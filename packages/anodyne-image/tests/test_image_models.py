from __future__ import annotations

import pytest
from anodyne_image.models import GeneratedImage, ImageManifestEntry, ImagePromptItem
from anodyne_image.ports import ImageProvider


def test_generated_image_defaults() -> None:
    img = GeneratedImage(data=b"abc")
    assert img.mime_type == "image/png"
    assert img.data == b"abc"


def test_prompt_item_label_optional() -> None:
    item = ImagePromptItem(item_index=3, prompt="a cat")
    assert item.label is None
    assert item.item_index == 3


def test_manifest_entry_roundtrip() -> None:
    entry = ImageManifestEntry(
        item_index=0, object_key="datasets/d/j/images/0.png", prompt="a cat", label="cat"
    )
    assert entry.mime_type == "image/png"
    dumped = entry.model_dump(mode="json")
    assert dumped["object_key"] == "datasets/d/j/images/0.png"


def test_image_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        ImageProvider()  # type: ignore[abstract]


async def test_image_provider_subclass_must_implement_generate() -> None:
    class _Concrete(ImageProvider):
        async def generate(
            self, prompt: str, *, seed: int, size: str = "1024x1024"
        ) -> GeneratedImage:
            return GeneratedImage(data=b"x")

    img = await _Concrete().generate("hi", seed=1)
    assert img.data == b"x"
