from uuid import uuid4

from anodyne_core.models import LLMRequest, Message, ModelConfig, Role, TenantContext, User


def test_tenant_context_has_role() -> None:
    u = User(id=uuid4(), tenant_id=uuid4(), subject="s", email="a@b.c", roles=[Role.ADMIN])
    ctx = TenantContext(tenant_id=u.tenant_id, user=u, roles=[Role.ADMIN])
    assert ctx.has_role(Role.ADMIN)
    assert not ctx.has_role(Role.OWNER)


def test_model_config_requires_secret_ref_for_cloud() -> None:
    m = ModelConfig(
        id=uuid4(),
        tenant_id=uuid4(),
        name="gpt",
        provider="openai",
        model="gpt-4o",
        secret_ref="ref-123",
    )
    assert m.enabled is True
    assert m.api_base is None


def test_llm_request_roundtrip() -> None:
    r = LLMRequest(model_config_id=uuid4(), messages=[Message(role="user", content="hi")])
    assert r.messages[0].role == "user"
