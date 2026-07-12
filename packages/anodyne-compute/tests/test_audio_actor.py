from __future__ import annotations

import pytest
import ray
from anodyne_compute.audio_actor import SelfHostedTTSActor

pytestmark = pytest.mark.integration


class _StubModel:
    def synthesize(self, text: str, voice: str | None) -> bytes:
        return f"{voice}:{text}".encode()


class _CountingModel:
    """Tags each instance with a fresh id so tests can detect re-construction."""

    _next_id = 0

    def __init__(self) -> None:
        _CountingModel._next_id += 1
        self._instance_id = _CountingModel._next_id

    def synthesize(self, text: str, voice: str | None) -> bytes:
        return f"{self._instance_id}:{text}".encode()


def test_actor_lazily_loads_model_and_synthesizes() -> None:
    ray.init(ignore_reinit_error=True)
    try:
        actor = SelfHostedTTSActor.remote(load_model=_StubModel)  # type: ignore[attr-defined]
        out = ray.get(actor.synthesize.remote("hello", "v1"))
        assert out == b"v1:hello"
    finally:
        ray.shutdown()


def test_actor_raises_without_a_configured_loader() -> None:
    ray.init(ignore_reinit_error=True)
    try:
        actor = SelfHostedTTSActor.remote()  # type: ignore[attr-defined]
        with pytest.raises(Exception, match="no model loader configured"):
            ray.get(actor.synthesize.remote("hi", None))
    finally:
        ray.shutdown()


def test_model_is_loaded_only_once_across_calls() -> None:
    # Both calls go to the same actor instance (a Ray actor is a single,
    # stateful process), so `_ensure_loaded` should construct `_CountingModel`
    # exactly once and reuse it -- verified by both calls reporting the same
    # `_instance_id`.
    ray.init(ignore_reinit_error=True)
    try:
        actor = SelfHostedTTSActor.remote(load_model=_CountingModel)  # type: ignore[attr-defined]
        first = ray.get(actor.synthesize.remote("a", None))
        second = ray.get(actor.synthesize.remote("b", None))
        assert first.split(b":")[0] == second.split(b":")[0]
    finally:
        ray.shutdown()
