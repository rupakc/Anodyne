"""`DatasetTemplate` -- a ready-made `DatasetSpec` blueprint for a common use-case.

Pure domain model (no generation logic); `catalog.py` holds the static catalog and the
`DatasetSpec` builder.
"""

from __future__ import annotations

from anodyne_dataset.models import FieldSpec, Modality
from pydantic import BaseModel, Field


class DatasetTemplate(BaseModel):
    key: str
    name: str
    description: str
    category: str
    modality: Modality = Modality.TABULAR
    fields: list[FieldSpec]
    default_target_rows: int
    default_directives: dict[str, object] = Field(default_factory=dict)
