from __future__ import annotations

import time
from collections.abc import AsyncIterator

import litellm
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider, SecretStore

__all__ = ["LiteLLMProvider", "litellm"]

# Drop provider-unsupported params instead of raising. Callers pass generation
# hints uniformly (e.g. `seed` for deterministic text generation), but not every
# provider accepts every param -- Gemini rejects `seed`, which otherwise raises
# litellm.UnsupportedParamsError and fails the whole call. With this flag litellm
# drops any param a given provider doesn't support, so the same request works
# across providers (the param simply has no effect where unsupported).
litellm.drop_params = True


class LiteLLMProvider(LLMProvider):
    def __init__(self, secret_store: SecretStore) -> None:
        self._secrets = secret_store

    def _kwargs(self, config: ModelConfig, request: LLMRequest) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": f"{config.provider}/{config.model}",
            "messages": [m.model_dump() for m in request.messages],
            **config.params,
            **request.params,
        }
        if config.secret_ref:
            kwargs["api_key"] = self._secrets.decrypt(config.secret_ref)
        if config.api_base:
            kwargs["api_base"] = config.api_base
        return kwargs

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        resp = await litellm.acompletion(**self._kwargs(config, request))
        latency = (time.perf_counter() - start) * 1000
        u = resp.usage
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            usage=Usage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
            ),
            cost=cost,
            latency_ms=latency,
        )

    async def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        resp = await litellm.acompletion(**self._kwargs(config, request), stream=True)
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
