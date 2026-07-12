"""Maps a `Profile` (inferred from an uploaded sample) to a reviewable `list[FieldSpec]`.

Mirrors `LLMSchemaProposer`'s role for the from-description path: the result flows through the
same `PATCH /datasets/{id}` review/edit endpoint C0 already has.
"""

from __future__ import annotations

from anodyne_dataset.models import FieldSpec, Profile, SemanticType


def fields_from_profile(profile: Profile) -> list[FieldSpec]:
    """Build a `FieldSpec` per profiled column, preserving column order."""
    fields = []
    for column in profile.columns:
        constraints: dict[str, object] = {}
        if column.semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT):
            if column.min is not None:
                constraints["min"] = column.min
            if column.max is not None:
                constraints["max"] = column.max
        elif column.semantic_type is SemanticType.CATEGORICAL and column.categories:
            constraints["choices"] = list(column.categories)
            constraints["most_frequent"] = max(column.categories, key=column.categories.get)  # type: ignore[arg-type]
        fields.append(
            FieldSpec(
                name=column.name,
                semantic_type=column.semantic_type,
                nullable=column.nullable,
                constraints=constraints,
            )
        )
    return fields
