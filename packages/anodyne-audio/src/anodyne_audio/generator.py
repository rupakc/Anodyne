from __future__ import annotations

import asyncio
from dataclasses import dataclass

from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult, DatasetSpec
from anodyne_dataset.ports import AudioProvider
from faker import Faker


@dataclass
class AudioItemPlan:
    """One planned audio item: its row index, the request to send the provider, and its label."""

    index: int
    request: AudioSynthesisRequest
    label: str | None


def _audio_directives(spec: DatasetSpec) -> dict[str, object]:
    raw = spec.directives.get("audio")
    return raw if isinstance(raw, dict) else {}


class AudioDatasetGenerator:
    """Orchestrates `AudioProvider` calls for a shard of a `Modality.AUDIO` DatasetSpec.

    Item text comes from `directives["audio"]["prompts"][i]` if provided (list
    index == row index), else a seeded, deterministic Faker sentence -- so
    "generate N audio items" works with zero directives, mirroring
    `TabularSampler`'s TEXT-field fallback (same seed + row index => same text).
    """

    def __init__(self, provider: AudioProvider) -> None:
        self._provider = provider

    def plan_items(
        self, spec: DatasetSpec, start_row: int, count: int, seed: int
    ) -> list[AudioItemPlan]:
        directives = _audio_directives(spec)
        prompts = directives.get("prompts")
        prompts = prompts if isinstance(prompts, list) else None
        labels = directives.get("labels")
        labels = labels if isinstance(labels, list) else None
        voice = directives.get("voice")
        voice = voice if isinstance(voice, str) else None
        language = directives.get("language")
        language = language if isinstance(language, str) else None

        plans: list[AudioItemPlan] = []
        for i in range(start_row, start_row + count):
            if prompts is not None and i < len(prompts):
                text = str(prompts[i])
            else:
                fake = Faker()
                Faker.seed(seed * 1_000_003 + i)
                text = fake.sentence()
            label = str(labels[i]) if labels is not None and i < len(labels) else None
            plans.append(
                AudioItemPlan(
                    index=i,
                    label=label,
                    request=AudioSynthesisRequest(text=text, voice=voice, language=language),
                )
            )
        return plans

    async def generate(
        self, spec: DatasetSpec, start_row: int, count: int, seed: int
    ) -> list[tuple[AudioItemPlan, AudioSynthesisResult]]:
        plans = self.plan_items(spec, start_row, count, seed)
        results = await asyncio.gather(*(self._provider.synthesize(p.request) for p in plans))
        return list(zip(plans, results, strict=True))
