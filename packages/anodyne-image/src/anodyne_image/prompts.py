from __future__ import annotations

from anodyne_dataset.models import DatasetSpec, SemanticType

from anodyne_image.models import ImagePromptItem

_DIRECTIVE_KEYS = ("bias", "use_case", "edge_case", "style")


class ImagePromptBuilder:
    """Derives deterministic, per-item generation prompts from a `DatasetSpec`.

    No randomness is involved: the label assigned to `item_index` is a pure
    function of the spec's label choices (`item_index % len(choices)`), and the
    prompt text is a pure function of the spec's description/label/directives.
    Same spec + range (regardless of how it's sharded) ⇒ identical prompts --
    the "deterministic-in-shape" requirement for a fully-mocked `ImageProvider`.
    """

    def build(self, spec: DatasetSpec, start_row: int, count: int) -> list[ImagePromptItem]:
        choices = self._label_choices(spec)
        items: list[ImagePromptItem] = []
        for offset in range(count):
            item_index = start_row + offset
            label = choices[item_index % len(choices)] if choices else None
            items.append(
                ImagePromptItem(
                    item_index=item_index, label=label, prompt=self._compose(spec, label)
                )
            )
        return items

    def _label_choices(self, spec: DatasetSpec) -> list[str]:
        for field in spec.fields:
            if field.semantic_type is SemanticType.CATEGORICAL:
                choices = field.constraints.get("choices")
                if isinstance(choices, list) and choices:
                    return [str(c) for c in choices]
        labels = spec.directives.get("labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        return []

    def _compose(self, spec: DatasetSpec, label: str | None) -> str:
        parts: list[str] = []
        description = spec.description.strip()
        if description:
            parts.append(description)
        if label:
            parts.append(f"class: {label}")
        for key in _DIRECTIVE_KEYS:
            value = spec.directives.get(key)
            if value:
                parts.append(f"{key.replace('_', '-')}: {value}")
        return ", ".join(parts) if parts else "a photorealistic image"
