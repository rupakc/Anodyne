from uuid import uuid4

import anodyne_llm.adapter as adapter_mod
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_llm.adapter import LiteLLMProvider


class _FakeSecrets(SecretStore):
    def encrypt(self, plaintext: str) -> str:
        return "ref"

    def decrypt(self, ref: str) -> str:
        return "sk-test-key"


async def test_complete_resolves_key_and_normalizes(monkeypatch):  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)

        class _Msg:
            content = "hi there"

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 3
            completion_tokens = 2
            total_tokens = 5

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()

    monkeypatch.setattr(adapter_mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(adapter_mod.litellm, "completion_cost", lambda completion_response: 0.01)

    cfg = ModelConfig(
        id=uuid4(),
        tenant_id=uuid4(),
        name="c",
        provider="openai",
        model="gpt-4o",
        secret_ref="ref",
    )
    req = LLMRequest(model_config_id=cfg.id, messages=[Message(role="user", content="hey")])
    resp = await LiteLLMProvider(_FakeSecrets()).complete(cfg, req)

    assert resp.content == "hi there"
    assert resp.usage.total_tokens == 5
    assert resp.cost == 0.01
    assert captured["model"] == "openai/gpt-4o"
    assert captured["api_key"] == "sk-test-key"
    assert captured["messages"] == [{"role": "user", "content": "hey"}]


def test_litellm_drops_provider_unsupported_params() -> None:
    # Regression: Gemini (and other providers) reject params they don't support
    # (e.g. `seed`), which raised litellm.UnsupportedParamsError and failed every
    # text-generation shard once Gemini became the default. Importing the adapter
    # must enable litellm's drop-unsupported-params behavior so such params are
    # silently dropped per-provider instead of crashing the call.
    assert adapter_mod.litellm.drop_params is True
