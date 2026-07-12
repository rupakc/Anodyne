"""Ray GPU actor wrapper for self-hosted OSS TTS/audio models (e.g. XTTS, Bark).

Real model loading/inference needs a GPU node and the target model package
(e.g. `coqui-tts` for XTTS) -- neither is available in this environment, so
loading is fully deferred to an injected `load_model` callable. Production
wiring constructs this actor with a loader that imports and loads the real
model onto GPU; nothing here imports a heavy ML package eagerly, so this
module is safe to import (and its Ray plumbing testable) without one.
"""

from __future__ import annotations

from collections.abc import Callable

import ray


@ray.remote
class SelfHostedTTSActor:
    def __init__(
        self, load_model: Callable[[], object] | None = None, model_name: str = "xtts_v2"
    ) -> None:
        self._model_name = model_name
        self._load_model = load_model
        self._model: object | None = None

    def _ensure_loaded(self) -> object:
        if self._model is None:
            if self._load_model is None:
                raise RuntimeError(
                    "no model loader configured; production wiring must inject one that "
                    f"loads {self._model_name} onto a GPU node (e.g. via coqui-tts's `TTS` API)"
                )
            self._model = self._load_model()
        return self._model

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        model = self._ensure_loaded()
        return model.synthesize(text, voice)  # type: ignore[no-any-return, attr-defined]
