from __future__ import annotations

from uuid import UUID

import jwt
import pytest
from anodyne_tenancy.oidc import AuthError, TokenValidator
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

TID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def keypair() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key: RSAPrivateKey, **overrides: object) -> str:
    claims: dict[str, object] = {
        "sub": "user-1",
        "email": "u@x.io",
        "org_id": TID,
        "aud": "anodyne",
        "iss": "https://kc/realms/anodyne",
        "realm_access": {"roles": ["admin", "irrelevant"]},
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


class _StubJWKS:
    def __init__(self, key: RSAPrivateKey) -> None:
        self._pub = key.public_key()

    def get_signing_key_from_jwt(self, token: str) -> object:  # mirrors PyJWKClient API
        class _K:
            pass

        k = _K()
        k.key = self._pub  # type: ignore[attr-defined]
        return k


def test_valid_token_yields_context(keypair: RSAPrivateKey) -> None:
    v = TokenValidator(_StubJWKS(keypair), issuer="https://kc/realms/anodyne", audience="anodyne")
    ctx = v.validate(_token(keypair))
    assert ctx.tenant_id == UUID(TID)
    assert ctx.user.email == "u@x.io"
    from anodyne_core.models import Role

    assert Role.ADMIN in ctx.roles


def test_missing_tenant_raises(keypair: RSAPrivateKey) -> None:
    v = TokenValidator(_StubJWKS(keypair), issuer="https://kc/realms/anodyne", audience="anodyne")
    with pytest.raises(AuthError):
        v.validate(_token(keypair, org_id=None))
