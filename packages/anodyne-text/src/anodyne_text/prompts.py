from __future__ import annotations

from anodyne_core.models import Message
from anodyne_dataset.models import DatasetSpec, FieldSpec

from anodyne_text.shapes import TextShape

_SHAPE_INSTRUCTIONS: dict[TextShape, str] = {
    TextShape.CLASSIFICATION: (
        "a text-classification corpus (short text samples paired with a label)"
    ),
    TextShape.QA: "a question-answering corpus (a question paired with a correct, concise answer)",
    TextShape.SUMMARIZATION: (
        "a summarization corpus (a longer document paired with a faithful summary)"
    ),
    TextShape.CHAT: (
        "an instruction/chat corpus (an instruction or user turn paired with a response)"
    ),
    TextShape.GENERIC: "a structured text corpus",
}


def _field_line(field: FieldSpec) -> str:
    choices = field.constraints.get("choices") if field.constraints else None
    if isinstance(choices, list) and choices:
        return f'  - "{field.name}": string, one of {choices!r}'
    return f'  - "{field.name}": string'


def build_batch_prompt(
    spec: DatasetSpec,
    shape: TextShape,
    batch_size: int,
    seed: int,
    batch_index: int,
) -> list[Message]:
    """Build the system/user message pair requesting one batch of rows.

    The system message pins down the exact JSON-array-of-objects contract
    (field names + types) and folds in `spec.description` plus any
    `spec.directives` steering (topic/tone/label-balance/etc, whatever the
    caller populated) as plain instructional text -- directives are a free-form
    dict on the C0 domain model, so no structured directive schema is assumed
    here beyond "stringify whatever's present".
    """
    field_lines = "\n".join(_field_line(f) for f in spec.fields)
    directive_lines = "\n".join(f"- {key}: {value}" for key, value in spec.directives.items())

    system_parts = [
        f"You generate {_SHAPE_INSTRUCTIONS[shape]} for synthetic dataset creation.",
        f"Dataset description: {spec.description}",
        "Return ONLY a JSON array of objects, each with exactly these fields:",
        field_lines,
        "No prose, no markdown fences unless you wrap the whole array in one ```json block.",
        "Every row must be realistic, non-empty, and distinct from the others in this batch.",
    ]
    if directive_lines:
        system_parts.append("Follow these additional steering directives:")
        system_parts.append(directive_lines)
    system = "\n".join(system_parts)

    user = (
        f"Generate {batch_size} rows for batch #{batch_index} "
        f"(seed={seed}). Return only the JSON array."
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]
