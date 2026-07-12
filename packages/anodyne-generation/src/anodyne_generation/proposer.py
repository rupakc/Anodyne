from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import FieldSpec
from anodyne_dataset.ports import SchemaProposer

if TYPE_CHECKING:
    pass


class SchemaProposalError(Exception):
    """Raised when schema proposal fails due to malformed output."""

    pass


_SYSTEM = (
    "You design tabular dataset schemas. Given a description, return ONLY a JSON array of "
    'fields: [{"name": str, "semantic_type": one of '
    "[integer,float,boolean,categorical,datetime,name,email,address,text], "
    '"nullable": bool (optional), "constraints": object (optional, e.g. {"min":0,"max":100} '
    'or {"choices":["a","b"]})}]. No prose.'
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMSchemaProposer(SchemaProposer):
    """Proposes a dataset schema using an LLM."""

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def propose(self, description: str) -> list[FieldSpec]:
        """Propose a schema from a natural language description.

        Args:
            description: Natural language description of the desired schema.

        Returns:
            List of FieldSpec objects representing the proposed schema.

        Raises:
            SchemaProposalError: If the LLM output cannot be parsed as a valid schema.
        """
        req = LLMRequest(
            model_config_id=self._cfg.id,
            messages=[
                Message(role="system", content=_SYSTEM),
                Message(role="user", content=description),
            ],
        )
        resp = await self._provider.complete(self._cfg, req)
        raw = resp.content.strip()
        m = _FENCE.search(raw)
        if m:
            raw = m.group(1).strip()
        try:
            data = json.loads(raw)
            return [FieldSpec.model_validate(item) for item in data]
        except Exception as exc:  # json/validation errors → domain error
            raise SchemaProposalError(f"could not parse schema from model output: {exc}") from exc
