from __future__ import annotations

from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_image.generator import ImageGenerator
from anodyne_image.models import GeneratedImage
from anodyne_image.ports import ImageProvider


class _FakeProvider(ImageProvider):
    """Deterministic stand-in: bytes are a pure function of prompt+seed, so
    the generator's output shape/content is verifiable without any network
    or GPU call.
    """

    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        return GeneratedImage(data=f"{prompt}|{seed}|{size}".encode())


class _RaisingProvider(ImageProvider):
    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        raise RuntimeError("boom")


def _spec(choices: list[str] | None = None, target_rows: int = 50) -> DatasetSpec:
    fields = (
        [
            FieldSpec(
                name="label",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": choices},
            )
        ]
        if choices
        else []
    )
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="a widget",
        modality=Modality.IMAGE,
        source="description",
        fields=fields,
        target_rows=target_rows,
    )


def test_generate_returns_expected_columns_and_row_count() -> None:
    spec = _spec(choices=["a", "b"])
    table = ImageGenerator(_FakeProvider()).generate(spec, 0, 10, seed=1)

    assert table.num_rows == 10
    assert set(table.column_names) == {"item_index", "label", "prompt", "image_bytes", "mime_type"}
    assert table.schema.field("image_bytes").type == pa.binary()


def test_deterministic_same_seed_and_range() -> None:
    spec = _spec(choices=["a", "b"])
    t1 = ImageGenerator(_FakeProvider()).generate(spec, 0, 10, seed=7)
    t2 = ImageGenerator(_FakeProvider()).generate(spec, 0, 10, seed=7)
    assert t1.equals(t2)


def test_disjoint_ranges_have_disjoint_item_indices() -> None:
    spec = _spec(choices=["a", "b"])
    generator = ImageGenerator(_FakeProvider())
    a = generator.generate(spec, 0, 5, seed=1).column("item_index").to_pylist()
    b = generator.generate(spec, 5, 5, seed=1).column("item_index").to_pylist()
    assert set(a).isdisjoint(set(b))
    assert a == [0, 1, 2, 3, 4]
    assert b == [5, 6, 7, 8, 9]


def test_image_bytes_reflect_prompt_and_per_item_seed() -> None:
    spec = _spec(choices=["a"])
    table = ImageGenerator(_FakeProvider()).generate(spec, 0, 2, seed=100)
    prompts = table.column("prompt").to_pylist()
    image_bytes = table.column("image_bytes").to_pylist()
    # seed passed to the provider is `seed + item_index`, so each item's bytes differ.
    assert image_bytes[0] == f"{prompts[0]}|100|1024x1024".encode()
    assert image_bytes[1] == f"{prompts[1]}|101|1024x1024".encode()


def test_provider_error_propagates() -> None:
    spec = _spec()
    try:
        ImageGenerator(_RaisingProvider()).generate(spec, 0, 1, seed=1)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")
