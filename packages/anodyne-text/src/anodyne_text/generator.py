from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_core.models import LLMRequest, ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import Generator

from anodyne_text.errors import TextGenerationError
from anodyne_text.prompts import build_batch_prompt
from anodyne_text.quality import Deduplicator, passes_quality
from anodyne_text.shapes import detect_shape, primary_field

_DEFAULT_BATCH_SIZE = 20
_DEFAULT_MAX_ATTEMPTS = 5
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_rows(content: str, field_names: list[str]) -> list[dict[str, str]]:
    """Parse an LLM batch response into validated rows.

    Tolerant of a fenced ```json block wrapping the array (mirrors
    `anodyne_generation.proposer.LLMSchemaProposer`'s extraction -- duplicated
    here, not imported, since `anodyne-generation` is intentionally left
    untouched by C2). Returns only rows that are objects containing every
    target field name as a string; malformed individual rows are dropped
    (not fatal), a wholly unparseable batch raises so the caller can retry.
    """
    raw = content.strip()
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TextGenerationError(f"could not parse a JSON array from model output: {exc}") from exc
    if not isinstance(data, list):
        raise TextGenerationError("model output was valid JSON but not an array of rows")

    rows: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if all(isinstance(item.get(name), str) for name in field_names):
            rows.append({name: item[name] for name in field_names})
    return rows


def _empty_table(field_names: list[str]) -> pa.Table:
    return pa.table({name: pa.array([], type=pa.string()) for name in field_names})


def _int_directive(directives: dict[str, object], key: str, default: int) -> int:
    value = directives.get(key, default) or default
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        return int(value)
    return default


class TextGenerator(Generator):
    """LLM-backed `Generator`: produces a text-corpus shard via batched, structured LLM calls.

    Constructed with the tenant's `LLMProvider` + `ModelConfig`, mirroring
    `anodyne_generation.proposer.LLMSchemaProposer`'s constructor shape.
    `generate` is synchronous per the `Generator` port; the async LLM calls
    are driven internally via `asyncio.run`, one shard-generation call at a
    time (this instance/call is not meant to be reused across concurrent
    event loops).
    """

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        field_names = [f.name for f in spec.fields]
        if count <= 0:
            return _empty_table(field_names)

        shape = detect_shape(spec.fields)
        primary = primary_field(shape, spec.fields)
        batch_size = _int_directive(spec.directives, "batch_size", _DEFAULT_BATCH_SIZE)
        max_attempts = _int_directive(spec.directives, "max_attempts", _DEFAULT_MAX_ATTEMPTS)

        dedup = Deduplicator()
        rows: list[dict[str, str]] = []
        for attempt in range(max_attempts):
            if len(rows) >= count:
                break
            remaining = count - len(rows)
            this_batch_size = min(batch_size, remaining)
            batch_index = start_row // max(batch_size, 1) + attempt
            messages = build_batch_prompt(spec, shape, this_batch_size, seed, batch_index)
            request = LLMRequest(
                model_config_id=self._cfg.id,
                messages=messages,
                params={"seed": seed},
            )
            response = asyncio.run(self._provider.complete(self._cfg, request))
            try:
                batch_rows = _extract_rows(response.content, field_names)
            except TextGenerationError:
                continue
            for row in batch_rows:
                if len(rows) >= count:
                    break
                row_any: dict[str, Any] = dict(row)
                if not passes_quality(row_any, primary):
                    continue
                if dedup.is_duplicate(row_any, primary):
                    continue
                rows.append(row)

        if not rows:
            raise TextGenerationError(
                f"no valid rows produced for dataset {spec.id} after {max_attempts} attempts"
            )

        return pa.table({name: [r[name] for r in rows] for name in field_names})
