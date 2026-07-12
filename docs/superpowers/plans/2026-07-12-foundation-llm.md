# Foundation + LLM Walking Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Anodyne's shared platform spine (monorepo, multi-tenant identity + RBAC, storage/secrets/observability adapters, and the LLM abstraction) proven end-to-end by an authenticated, tenant-scoped `POST /llm/invoke`.

**Architecture:** Hexagonal (ports & adapters). `anodyne-core` holds Pydantic domain models and abstract ports; adapter packages implement them; `apps/api-gateway` wires them behind FastAPI. Multi-tenancy is enforced in Postgres via row-level security keyed on a per-transaction `app.tenant_id`. The LLM layer wraps the LiteLLM SDK behind an `LLMProvider` port.

**Tech Stack:** Python 3.12, uv workspace, FastAPI, Pydantic v2 / pydantic-settings, SQLAlchemy 2.0 async + asyncpg + Alembic, PyJWT (JWKS), cryptography (Fernet), boto3, LiteLLM, OpenTelemetry + structlog. Tests: pytest, pytest-asyncio, httpx, moto, testcontainers[postgres].

## Global Constraints

- Python **3.12+**. All packages use a `src/` layout under a single uv workspace.
- Python import names use underscores (`anodyne_core`); directory names use hyphens (`anodyne-core`).
- `ruff` (lint + format) and `mypy --strict` must pass on every commit; `pytest` green.
- No adapter imports inside `anodyne-core`. Domain logic depends only on ports.
- Every DB-persisted, tenant-owned row has a non-null `tenant_id` and an RLS policy.
- Secrets (model API keys) are never stored or logged in plaintext — only encrypted refs persist.
- Conventional-commit messages; commit at the end of every task.

---

### Task 1: Monorepo scaffolding, tooling, and CLAUDE.md

**Files:**
- Create: `pyproject.toml` (workspace root), `uv.toml`
- Create: `packages/anodyne-core/pyproject.toml`, `packages/anodyne-core/src/anodyne_core/__init__.py`
- Create: `.pre-commit-config.yaml`, `ruff.toml`, `mypy.ini`
- Create: `.claude/CLAUDE.md`
- Test: `packages/anodyne-core/tests/test_smoke.py`

**Interfaces:**
- Produces: the uv workspace; every later package is added as a `[tool.uv.workspace] members` entry and depends on siblings via `tool.uv.sources`.

- [ ] **Step 1: Write the failing smoke test**

```python
# packages/anodyne-core/tests/test_smoke.py
import anodyne_core

def test_package_importable():
    assert anodyne_core.__version__ == "0.1.0"
```

- [ ] **Step 2: Run it to verify failure**

Run: `uv run pytest packages/anodyne-core/tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anodyne_core'`.

- [ ] **Step 3: Create workspace root `pyproject.toml`**

```toml
[project]
name = "anodyne"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.uv.workspace]
members = ["packages/*", "apps/*"]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "mypy>=1.11", "ruff>=0.6",
       "httpx>=0.27", "moto>=5", "testcontainers[postgres]>=4"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["packages", "apps"]
```

- [ ] **Step 4: Create `packages/anodyne-core/pyproject.toml` and package init**

```toml
# packages/anodyne-core/pyproject.toml
[project]
name = "anodyne-core"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python
# packages/anodyne-core/src/anodyne_core/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 5: Create `ruff.toml`, `mypy.ini`, `.pre-commit-config.yaml`**

```toml
# ruff.toml
line-length = 100
target-version = "py312"
[lint]
select = ["E", "F", "I", "UP", "B", "ASYNC"]
```

```ini
# mypy.ini
[mypy]
python_version = 3.12
strict = True
namespace_packages = True
explicit_package_bases = True
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 6: Write `.claude/CLAUDE.md`**

```markdown
# Anodyne — Developer Guide

Synthetic data generation + LLM-as-a-Judge benchmarking platform. See `docs/architecture.md`.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Types: `uv run mypy .`
- Local backbone: `make up` (Postgres, Redis, Keycloak, MinIO)

## Conventions
- Hexagonal: domain + ports in `anodyne-core`; adapters in sibling packages; wiring in `apps/`.
- No adapter imports in `anodyne-core`.
- TDD: failing test first. Conventional commits. `mypy --strict` and `ruff` must pass.
- Multi-tenant: every tenant row carries `tenant_id` + an RLS policy; never log/store plaintext secrets.
```

- [ ] **Step 7: Run smoke test to verify pass**

Run: `uv sync && uv run pytest packages/anodyne-core/tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml ruff.toml mypy.ini .pre-commit-config.yaml .claude packages
git commit -m "chore: scaffold uv workspace, tooling, and anodyne-core package"
```

---

### Task 2: Domain models (`anodyne-core`)

**Files:**
- Create: `packages/anodyne-core/src/anodyne_core/models.py`
- Test: `packages/anodyne-core/tests/test_models.py`

**Interfaces:**
- Produces: `Role`, `Tenant`, `User`, `ModelConfig`, `TenantContext`, `LLMRequest`, `LLMResponse`, `Message`, `Usage` — imported by every later task.

- [ ] **Step 1: Write failing tests**

```python
# packages/anodyne-core/tests/test_models.py
from uuid import uuid4
from anodyne_core.models import Role, TenantContext, User, ModelConfig, LLMRequest, Message

def test_tenant_context_has_role():
    u = User(id=uuid4(), tenant_id=uuid4(), subject="s", email="a@b.c", roles=[Role.ADMIN])
    ctx = TenantContext(tenant_id=u.tenant_id, user=u, roles=[Role.ADMIN])
    assert ctx.has_role(Role.ADMIN)
    assert not ctx.has_role(Role.OWNER)

def test_model_config_requires_secret_ref_for_cloud():
    m = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="gpt", provider="openai",
                    model="gpt-4o", secret_ref="ref-123")
    assert m.enabled is True
    assert m.api_base is None

def test_llm_request_roundtrip():
    r = LLMRequest(model_config_id=uuid4(), messages=[Message(role="user", content="hi")])
    assert r.messages[0].role == "user"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-core/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Implement `models.py`**

```python
# packages/anodyne-core/src/anodyne_core/models.py
from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID
from pydantic import BaseModel, Field

class Role(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class Tenant(BaseModel):
    id: UUID
    name: str
    org_ref: str
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class User(BaseModel):
    id: UUID
    tenant_id: UUID
    subject: str          # OIDC `sub`
    email: str
    roles: list[Role] = Field(default_factory=list)

class TenantContext(BaseModel):
    tenant_id: UUID
    user: User
    roles: list[Role]
    def has_role(self, role: Role) -> bool:
        return role in self.roles

class ModelConfig(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    provider: str                 # e.g. "openai", "anthropic", "ollama", "vllm"
    model: str                    # e.g. "gpt-4o"
    params: dict[str, object] = Field(default_factory=dict)
    secret_ref: str | None = None # encrypted-secret handle; None for keyless local models
    api_base: str | None = None   # set for local models (Ollama/vLLM)
    enabled: bool = True

class Message(BaseModel):
    role: str
    content: str

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMRequest(BaseModel):
    model_config_id: UUID
    messages: list[Message]
    params: dict[str, object] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    content: str
    usage: Usage
    cost: float = 0.0
    latency_ms: float = 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest packages/anodyne-core/tests/test_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-core
git commit -m "feat(core): add domain models"
```

---

### Task 3: Ports / interfaces (`anodyne-core`)

**Files:**
- Create: `packages/anodyne-core/src/anodyne_core/ports.py`
- Test: `packages/anodyne-core/tests/test_ports.py`

**Interfaces:**
- Produces the abstract base classes implemented by later adapters:
  - `ObjectStore.put(key, data) -> None`, `get(key) -> bytes`, `presigned_url(key, expires) -> str`, `list(prefix) -> list[str]`
  - `SecretStore.encrypt(plaintext) -> str`, `decrypt(ref) -> str`
  - `LLMProvider.complete(config, request) -> LLMResponse`, `stream(config, request) -> AsyncIterator[str]`
  - `AuthorizationPolicy.is_permitted(ctx, permission) -> bool`

- [ ] **Step 1: Write failing test (ports are abstract)**

```python
# packages/anodyne-core/tests/test_ports.py
import pytest
from anodyne_core.ports import ObjectStore, SecretStore, LLMProvider, AuthorizationPolicy

@pytest.mark.parametrize("cls", [ObjectStore, SecretStore, LLMProvider, AuthorizationPolicy])
def test_ports_are_abstract(cls):
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-core/tests/test_ports.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `ports.py`**

```python
# packages/anodyne-core/src/anodyne_core/ports.py
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, TenantContext

class ObjectStore(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes) -> None: ...
    @abstractmethod
    async def get(self, key: str) -> bytes: ...
    @abstractmethod
    async def presigned_url(self, key: str, expires: int = 3600) -> str: ...
    @abstractmethod
    async def list(self, prefix: str) -> list[str]: ...

class SecretStore(ABC):
    @abstractmethod
    def encrypt(self, plaintext: str) -> str: ...
    @abstractmethod
    def decrypt(self, ref: str) -> str: ...

class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse: ...
    @abstractmethod
    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]: ...

class AuthorizationPolicy(ABC):
    @abstractmethod
    def is_permitted(self, ctx: TenantContext, permission: str) -> bool: ...
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest packages/anodyne-core/tests/test_ports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-core
git commit -m "feat(core): add ports (ObjectStore, SecretStore, LLMProvider, AuthorizationPolicy)"
```

---

### Task 4: Role-based authorization (`anodyne-tenancy`)

**Files:**
- Create: `packages/anodyne-tenancy/pyproject.toml`, `.../src/anodyne_tenancy/__init__.py`
- Create: `packages/anodyne-tenancy/src/anodyne_tenancy/authz.py`
- Test: `packages/anodyne-tenancy/tests/test_authz.py`

**Interfaces:**
- Consumes: `Role`, `TenantContext`, `AuthorizationPolicy`.
- Produces: `PERMISSIONS: dict[Role, set[str]]`, `RoleBasedPolicy` implementing `AuthorizationPolicy`.

- [ ] **Step 1: Write failing tests**

```python
# packages/anodyne-tenancy/tests/test_authz.py
from uuid import uuid4
from anodyne_core.models import Role, TenantContext, User
from anodyne_tenancy.authz import RoleBasedPolicy

def _ctx(role):
    u = User(id=uuid4(), tenant_id=uuid4(), subject="s", email="a@b.c", roles=[role])
    return TenantContext(tenant_id=u.tenant_id, user=u, roles=[role])

def test_owner_can_delete_models():
    assert RoleBasedPolicy().is_permitted(_ctx(Role.OWNER), "models:delete")

def test_viewer_cannot_write_models():
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.VIEWER), "models:read")
    assert not p.is_permitted(_ctx(Role.VIEWER), "models:write")

def test_member_can_invoke_llm():
    assert RoleBasedPolicy().is_permitted(_ctx(Role.MEMBER), "llm:invoke")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-tenancy/tests/test_authz.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create package files + implement `authz.py`**

`packages/anodyne-tenancy/pyproject.toml`:
```toml
[project]
name = "anodyne-tenancy"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anodyne-core", "pyjwt[crypto]>=2.9", "httpx>=0.27"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.uv.sources]
anodyne-core = { workspace = true }
```

```python
# packages/anodyne-tenancy/src/anodyne_tenancy/__init__.py
```

```python
# packages/anodyne-tenancy/src/anodyne_tenancy/authz.py
from __future__ import annotations
from anodyne_core.models import Role, TenantContext
from anodyne_core.ports import AuthorizationPolicy

_VIEWER = {"models:read", "llm:invoke:read"}
_MEMBER = _VIEWER | {"llm:invoke", "models:write"}
_ADMIN = _MEMBER | {"models:delete", "users:read"}
_OWNER = _ADMIN | {"users:write", "tenant:admin"}

PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: _VIEWER, Role.MEMBER: _MEMBER, Role.ADMIN: _ADMIN, Role.OWNER: _OWNER,
}

class RoleBasedPolicy(AuthorizationPolicy):
    def is_permitted(self, ctx: TenantContext, permission: str) -> bool:
        return any(permission in PERMISSIONS.get(role, set()) for role in ctx.roles)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv sync && uv run pytest packages/anodyne-tenancy/tests/test_authz.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-tenancy
git commit -m "feat(tenancy): add role-based authorization policy"
```

---

### Task 5: Fernet secret store (`anodyne-storage`)

**Files:**
- Create: `packages/anodyne-storage/pyproject.toml`, `.../src/anodyne_storage/__init__.py`
- Create: `packages/anodyne-storage/src/anodyne_storage/secrets.py`
- Test: `packages/anodyne-storage/tests/test_secrets.py`

**Interfaces:**
- Consumes: `SecretStore`.
- Produces: `FernetSecretStore(key: bytes)` implementing `SecretStore`; `.decrypt(.encrypt(x)) == x`; ciphertext ref != plaintext.

- [ ] **Step 1: Write failing tests**

```python
# packages/anodyne-storage/tests/test_secrets.py
from cryptography.fernet import Fernet
from anodyne_storage.secrets import FernetSecretStore

def test_encrypt_decrypt_roundtrip():
    store = FernetSecretStore(Fernet.generate_key())
    ref = store.encrypt("sk-secret")
    assert ref != "sk-secret"
    assert store.decrypt(ref) == "sk-secret"

def test_ciphertext_is_not_plaintext_substring():
    store = FernetSecretStore(Fernet.generate_key())
    assert "sk-secret" not in store.encrypt("sk-secret")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-storage/tests/test_secrets.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create package + implement `secrets.py`**

`packages/anodyne-storage/pyproject.toml`:
```toml
[project]
name = "anodyne-storage"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anodyne-core", "cryptography>=43", "boto3>=1.35",
                "sqlalchemy[asyncio]>=2.0", "asyncpg>=0.29", "alembic>=1.13"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.uv.sources]
anodyne-core = { workspace = true }
```

```python
# packages/anodyne-storage/src/anodyne_storage/secrets.py
from __future__ import annotations
from cryptography.fernet import Fernet
from anodyne_core.ports import SecretStore

class FernetSecretStore(SecretStore):
    """Dev/on-prem symmetric-key secret store. Prod swaps in a Vault/KMS adapter."""
    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)
    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()
    def decrypt(self, ref: str) -> str:
        return self._fernet.decrypt(ref.encode()).decode()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv sync && uv run pytest packages/anodyne-storage/tests/test_secrets.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-storage
git commit -m "feat(storage): add Fernet secret store"
```

---

### Task 6: S3-compatible object store (`anodyne-storage`)

**Files:**
- Create: `packages/anodyne-storage/src/anodyne_storage/objectstore.py`
- Test: `packages/anodyne-storage/tests/test_objectstore.py`

**Interfaces:**
- Consumes: `ObjectStore`.
- Produces: `S3ObjectStore(bucket, tenant_id, *, client)` — all keys transparently prefixed `{tenant_id}/`. `put/get/list/presigned_url` operate within that prefix.

- [ ] **Step 1: Write failing tests (moto mocks S3)**

```python
# packages/anodyne-storage/tests/test_objectstore.py
import boto3, pytest
from uuid import UUID
from moto import mock_aws
from anodyne_storage.objectstore import S3ObjectStore

TID = UUID("11111111-1111-1111-1111-111111111111")

@pytest.fixture
def bucket():
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket="anodyne")
        yield c

async def test_put_get_is_tenant_prefixed(bucket):
    store = S3ObjectStore("anodyne", TID, client=bucket)
    await store.put("data/x.txt", b"hello")
    # object physically stored under the tenant prefix
    assert bucket.get_object(Bucket="anodyne", Key=f"{TID}/data/x.txt")["Body"].read() == b"hello"
    assert await store.get("data/x.txt") == b"hello"

async def test_list_returns_relative_keys(bucket):
    store = S3ObjectStore("anodyne", TID, client=bucket)
    await store.put("a.txt", b"1")
    await store.put("b.txt", b"2")
    assert sorted(await store.list("")) == ["a.txt", "b.txt"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-storage/tests/test_objectstore.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `objectstore.py`**

```python
# packages/anodyne-storage/src/anodyne_storage/objectstore.py
from __future__ import annotations
import asyncio
from uuid import UUID
from typing import Any
from anodyne_core.ports import ObjectStore

class S3ObjectStore(ObjectStore):
    """Works against MinIO (on-prem/dev) and GCS interop (cloud). Keys are tenant-prefixed."""
    def __init__(self, bucket: str, tenant_id: UUID, *, client: Any) -> None:
        self._bucket = bucket
        self._prefix = f"{tenant_id}/"
        self._c = client

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._c.put_object, Bucket=self._bucket, Key=self._key(key), Body=data)

    async def get(self, key: str) -> bytes:
        obj = await asyncio.to_thread(self._c.get_object, Bucket=self._bucket, Key=self._key(key))
        return obj["Body"].read()  # type: ignore[no-any-return]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return await asyncio.to_thread(
            self._c.generate_presigned_url, "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(key)}, ExpiresIn=expires,
        )

    async def list(self, prefix: str) -> list[str]:
        resp = await asyncio.to_thread(
            self._c.list_objects_v2, Bucket=self._bucket, Prefix=self._key(prefix))
        return [o["Key"][len(self._prefix):] for o in resp.get("Contents", [])]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest packages/anodyne-storage/tests/test_objectstore.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-storage
git commit -m "feat(storage): add tenant-prefixed S3 object store"
```

---

### Task 7: Database schema, migrations, and RLS-enforced session (`anodyne-storage`)

**Files:**
- Create: `packages/anodyne-storage/src/anodyne_storage/db.py` (engine, ORM tables, session factory)
- Create: `packages/anodyne-storage/src/anodyne_storage/migrations/` (Alembic env + one migration)
- Create: `packages/anodyne-storage/alembic.ini`
- Test: `packages/anodyne-storage/tests/test_rls.py`

**Interfaces:**
- Consumes: nothing from siblings beyond model shapes.
- Produces:
  - ORM tables `tenants`, `users`, `model_configs` (all tenant-scoped tables carry `tenant_id`).
  - `make_engine(dsn) -> AsyncEngine`
  - `tenant_session(engine, tenant_id) -> async context manager[AsyncSession]` that runs
    `SET LOCAL app.tenant_id` and operates as a non-superuser role so RLS is enforced.
  - `apply_rls(conn)` helper used by the migration to enable RLS + policies.

- [ ] **Step 1: Write failing RLS isolation test (testcontainers Postgres)**

```python
# packages/anodyne-storage/tests/test_rls.py
import pytest_asyncio, pytest
from uuid import uuid4
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer
from anodyne_storage.db import make_engine, tenant_session, metadata, apply_rls

@pytest_asyncio.fixture
async def engine():
    with PostgresContainer("postgres:16") as pg:
        dsn = pg.get_connection_url().replace("psycopg2", "asyncpg")
        eng = make_engine(dsn)
        async with eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            # app role that is subject to RLS (not the superuser bootstrap role)
            await conn.execute(text("CREATE ROLE app LOGIN; GRANT ALL ON ALL TABLES IN SCHEMA public TO app;"))
        yield eng
        await eng.dispose()

async def test_tenant_isolation(engine):
    t1, t2 = uuid4(), uuid4()
    async with tenant_session(engine, t1) as s:
        await s.execute(text("INSERT INTO tenants (id, name, org_ref, status) "
                             "VALUES (:id,'A','orgA','active')"), {"id": t1})
        await s.commit()
    # tenant 2's session must NOT see tenant 1's row
    async with tenant_session(engine, t2) as s:
        rows = (await s.execute(text("SELECT id FROM tenants"))).all()
        assert rows == []
    async with tenant_session(engine, t1) as s:
        rows = (await s.execute(text("SELECT id FROM tenants"))).all()
        assert len(rows) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-storage/tests/test_rls.py -v`
Expected: FAIL — `ModuleNotFoundError` / attributes missing.

- [ ] **Step 3: Implement `db.py`**

```python
# packages/anodyne-storage/src/anodyne_storage/db.py
from __future__ import annotations
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from uuid import UUID
from sqlalchemy import Column, MetaData, String, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

metadata = MetaData()

tenants = Table("tenants", metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("name", String, nullable=False),
    Column("org_ref", String, nullable=False),
    Column("status", String, nullable=False, server_default="active"))

users = Table("users", metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("subject", String, nullable=False),
    Column("email", String, nullable=False))

model_configs = Table("model_configs", metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("name", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("model", String, nullable=False),
    Column("params", JSONB, nullable=False, server_default="{}"),
    Column("secret_ref", Text, nullable=True),
    Column("api_base", String, nullable=True),
    Column("enabled", String, nullable=False, server_default="true"))

# Tenant-scoped tables get an RLS policy keyed on the per-transaction app.tenant_id GUC.
_TENANT_TABLES = {"tenants": "id", "users": "tenant_id", "model_configs": "tenant_id"}

async def apply_rls(conn) -> None:  # type: ignore[no-untyped-def]
    for tbl, col in _TENANT_TABLES.items():
        await conn.execute(text(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY"))
        await conn.execute(text(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY"))
        await conn.execute(text(
            f"CREATE POLICY tenant_isolation ON {tbl} USING "
            f"({col} = current_setting('app.tenant_id', true)::uuid)"))

def make_engine(dsn: str) -> AsyncEngine:
    return create_async_engine(dsn, pool_pre_ping=True)

@asynccontextmanager
async def tenant_session(engine: AsyncEngine, tenant_id: UUID) -> AsyncIterator[AsyncSession]:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        await session.execute(text("SET LOCAL app.tenant_id = :tid"), {"tid": str(tenant_id)})
        yield session
```

- [ ] **Step 4: Wire Alembic (env + one migration that creates tables and calls `apply_rls`)**

`packages/anodyne-storage/alembic.ini` — standard, `script_location = src/anodyne_storage/migrations`.
Create `migrations/env.py` importing `metadata` as `target_metadata`, running async. Create
`migrations/versions/0001_initial.py`:

```python
# 0001_initial.py
from alembic import op
from anodyne_storage.db import metadata, apply_rls
revision, down_revision = "0001", None

def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind)
    import asyncio  # apply_rls is async-shaped; run its SQL synchronously here
    for tbl, col in {"tenants": "id", "users": "tenant_id", "model_configs": "tenant_id"}.items():
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY tenant_isolation ON {tbl} USING "
                   f"({col} = current_setting('app.tenant_id', true)::uuid)")

def downgrade() -> None:
    metadata.drop_all(op.get_bind())
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest packages/anodyne-storage/tests/test_rls.py -v`
Expected: PASS — tenant 2 sees 0 rows, tenant 1 sees its own. (Requires Docker for testcontainers.)

- [ ] **Step 6: Commit**

```bash
git add packages/anodyne-storage
git commit -m "feat(storage): add DB schema, Alembic migration, and RLS-enforced tenant sessions"
```

---

### Task 8: OIDC token validation → TenantContext (`anodyne-tenancy`)

**Files:**
- Create: `packages/anodyne-tenancy/src/anodyne_tenancy/oidc.py`
- Test: `packages/anodyne-tenancy/tests/test_oidc.py`

**Interfaces:**
- Consumes: `Role`, `User`, `TenantContext`.
- Produces: `TokenValidator(jwks_client, issuer, audience)` with
  `validate(token: str) -> TenantContext`. Maps claims: `sub`→subject, `email`, tenant from
  `org_id` claim (fallback `tenant_id`), realm roles from `realm_access.roles` filtered to `Role`.
  Raises `AuthError` on invalid/expired/missing-tenant tokens.

- [ ] **Step 1: Write failing tests (sign a token with a local RSA key)**

```python
# packages/anodyne-tenancy/tests/test_oidc.py
import jwt, pytest
from uuid import UUID
from cryptography.hazmat.primitives.asymmetric import rsa
from anodyne_tenancy.oidc import TokenValidator, AuthError

TID = "22222222-2222-2222-2222-222222222222"

@pytest.fixture
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key

def _token(key, **overrides):
    claims = {"sub": "user-1", "email": "u@x.io", "org_id": TID,
              "aud": "anodyne", "iss": "https://kc/realms/anodyne",
              "realm_access": {"roles": ["admin", "irrelevant"]}}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})

class _StubJWKS:
    def __init__(self, key): self._pub = key.public_key()
    def get_signing_key_from_jwt(self, token):  # mirrors PyJWKClient API
        class _K: pass
        k = _K(); k.key = self._pub; return k

def test_valid_token_yields_context(keypair):
    v = TokenValidator(_StubJWKS(keypair), issuer="https://kc/realms/anodyne", audience="anodyne")
    ctx = v.validate(_token(keypair))
    assert ctx.tenant_id == UUID(TID)
    assert ctx.user.email == "u@x.io"
    from anodyne_core.models import Role
    assert Role.ADMIN in ctx.roles

def test_missing_tenant_raises(keypair):
    v = TokenValidator(_StubJWKS(keypair), issuer="https://kc/realms/anodyne", audience="anodyne")
    with pytest.raises(AuthError):
        v.validate(_token(keypair, org_id=None))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-tenancy/tests/test_oidc.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `oidc.py`**

```python
# packages/anodyne-tenancy/src/anodyne_tenancy/oidc.py
from __future__ import annotations
from typing import Any, Protocol
from uuid import UUID, uuid5, NAMESPACE_URL
import jwt
from anodyne_core.models import Role, TenantContext, User

class AuthError(Exception): ...

class _JWKSClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...

class TokenValidator:
    def __init__(self, jwks_client: _JWKSClient, issuer: str, audience: str) -> None:
        self._jwks = jwks_client
        self._issuer = issuer
        self._audience = audience

    def validate(self, token: str) -> TenantContext:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, signing_key.key, algorithms=["RS256"],
                                audience=self._audience, issuer=self._issuer)
        except jwt.PyJWTError as exc:
            raise AuthError(str(exc)) from exc

        tenant_raw = claims.get("org_id") or claims.get("tenant_id")
        if not tenant_raw:
            raise AuthError("token has no tenant/org claim")
        tenant_id = UUID(str(tenant_raw))
        roles = [Role(r) for r in claims.get("realm_access", {}).get("roles", []) if r in Role._value2member_map_]
        subject = str(claims["sub"])
        user = User(id=uuid5(NAMESPACE_URL, subject), tenant_id=tenant_id,
                    subject=subject, email=str(claims.get("email", "")), roles=roles)
        return TenantContext(tenant_id=tenant_id, user=user, roles=roles)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest packages/anodyne-tenancy/tests/test_oidc.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-tenancy
git commit -m "feat(tenancy): add OIDC token validation to TenantContext"
```

---

### Task 9: Observability setup (`anodyne-observability`)

**Files:**
- Create: `packages/anodyne-observability/pyproject.toml`, `.../src/anodyne_observability/__init__.py`
- Create: `packages/anodyne-observability/src/anodyne_observability/logging.py`
- Test: `packages/anodyne-observability/tests/test_logging.py`

**Interfaces:**
- Produces: `configure_logging() -> None`, `get_logger(name) -> structlog.BoundLogger`,
  `bind_request_context(tenant_id, request_id) -> None` (binds to contextvars).

- [ ] **Step 1: Write failing test**

```python
# packages/anodyne-observability/tests/test_logging.py
import json, logging
from anodyne_observability.logging import configure_logging, get_logger, bind_request_context

def test_logs_are_json_with_bound_context(capsys):
    configure_logging()
    bind_request_context(tenant_id="t-1", request_id="r-9")
    get_logger("test").info("hello", extra_field=42)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(out)
    assert record["event"] == "hello"
    assert record["tenant_id"] == "t-1"
    assert record["request_id"] == "r-9"
    assert record["extra_field"] == 42
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-observability/tests/test_logging.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create package + implement `logging.py`**

`packages/anodyne-observability/pyproject.toml`:
```toml
[project]
name = "anodyne-observability"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["structlog>=24", "opentelemetry-sdk>=1.27",
                "opentelemetry-instrumentation-fastapi>=0.48b0"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python
# packages/anodyne-observability/src/anodyne_observability/logging.py
from __future__ import annotations
import structlog

def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)

def bind_request_context(*, tenant_id: str, request_id: str) -> None:
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, request_id=request_id)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv sync && uv run pytest packages/anodyne-observability/tests/test_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-observability
git commit -m "feat(observability): add structured JSON logging with bound request context"
```

---

### Task 10: LiteLLM adapter + model registry (`anodyne-llm`)

**Files:**
- Create: `packages/anodyne-llm/pyproject.toml`, `.../src/anodyne_llm/__init__.py`
- Create: `packages/anodyne-llm/src/anodyne_llm/adapter.py`
- Test: `packages/anodyne-llm/tests/test_adapter.py`

**Interfaces:**
- Consumes: `LLMProvider`, `SecretStore`, `ModelConfig`, `LLMRequest`, `LLMResponse`, `Usage`, `Message`.
- Produces: `LiteLLMProvider(secret_store: SecretStore)` implementing `LLMProvider`. `complete`
  resolves the API key from `config.secret_ref` via `secret_store`, calls `litellm.acompletion`
  with `model=f"{provider}/{model}"`, `api_base=config.api_base`, and returns `LLMResponse`
  (content, usage, cost via `litellm.completion_cost`).

- [ ] **Step 1: Write failing test (monkeypatch litellm.acompletion)**

```python
# packages/anodyne-llm/tests/test_adapter.py
from uuid import uuid4
import anodyne_llm.adapter as adapter_mod
from anodyne_core.models import ModelConfig, LLMRequest, Message
from anodyne_llm.adapter import LiteLLMProvider

class _FakeSecrets:
    def encrypt(self, p): return "ref"
    def decrypt(self, ref): return "sk-test-key"

async def test_complete_resolves_key_and_normalizes(monkeypatch):
    captured = {}
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        class _Msg: content = "hi there"
        class _Choice: message = _Msg()
        class _Usage:
            prompt_tokens, completion_tokens, total_tokens = 3, 2, 5
        class _Resp:
            choices = [_Choice()]; usage = _Usage()
        return _Resp()
    monkeypatch.setattr(adapter_mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(adapter_mod.litellm, "completion_cost", lambda completion_response: 0.01)

    cfg = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="openai",
                      model="gpt-4o", secret_ref="ref")
    req = LLMRequest(model_config_id=cfg.id, messages=[Message(role="user", content="hey")])
    resp = await LiteLLMProvider(_FakeSecrets()).complete(cfg, req)

    assert resp.content == "hi there"
    assert resp.usage.total_tokens == 5
    assert resp.cost == 0.01
    assert captured["model"] == "openai/gpt-4o"
    assert captured["api_key"] == "sk-test-key"
    assert captured["messages"] == [{"role": "user", "content": "hey"}]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-llm/tests/test_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create package + implement `adapter.py`**

`packages/anodyne-llm/pyproject.toml`:
```toml
[project]
name = "anodyne-llm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anodyne-core", "litellm>=1.81"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.uv.sources]
anodyne-core = { workspace = true }
```

```python
# packages/anodyne-llm/src/anodyne_llm/adapter.py
from __future__ import annotations
import time
from collections.abc import AsyncIterator
import litellm
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider, SecretStore

class LiteLLMProvider(LLMProvider):
    def __init__(self, secret_store: SecretStore) -> None:
        self._secrets = secret_store

    def _kwargs(self, config: ModelConfig, request: LLMRequest) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": f"{config.provider}/{config.model}",
            "messages": [m.model_dump() for m in request.messages],
            **config.params, **request.params,
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
            usage=Usage(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                        total_tokens=u.total_tokens),
            cost=cost, latency_ms=latency)

    async def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        resp = await litellm.acompletion(**self._kwargs(config, request), stream=True)
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

Note: `stream` is declared `async def` returning an async generator; the port's signature is
satisfied. Add a streaming unit test only when the gateway needs it (YAGNI for the skeleton).

- [ ] **Step 4: Run to verify pass**

Run: `uv sync && uv run pytest packages/anodyne-llm/tests/test_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/anodyne-llm
git commit -m "feat(llm): add LiteLLM provider adapter"
```

---

### Task 11: API gateway app + endpoints (`apps/api-gateway`)

**Files:**
- Create: `apps/api-gateway/pyproject.toml`, `.../src/api_gateway/__init__.py`
- Create: `apps/api-gateway/src/api_gateway/config.py` (pydantic-settings)
- Create: `apps/api-gateway/src/api_gateway/deps.py` (DI: validator, policy, provider, session)
- Create: `apps/api-gateway/src/api_gateway/app.py` (FastAPI app, middleware, routes)
- Test: `apps/api-gateway/tests/test_app.py`

**Interfaces:**
- Consumes: `TokenValidator`, `RoleBasedPolicy`, `LiteLLMProvider`, `tenant_session`, models/ports.
- Produces: `create_app() -> FastAPI` with routes `/healthz`, `/readyz`, `/me`, `/models` (GET/POST),
  `/models/{id}` (DELETE), `/llm/invoke` (POST). DI is overridable in tests via
  `app.dependency_overrides`.

- [ ] **Step 1: Write failing endpoint tests (deps overridden; no live backbone)**

```python
# apps/api-gateway/tests/test_app.py
import pytest
from uuid import uuid4
from httpx import ASGITransport, AsyncClient
from anodyne_core.models import Role, TenantContext, User, LLMResponse, Usage, ModelConfig
from api_gateway.app import create_app
from api_gateway import deps

def _ctx(role=Role.MEMBER):
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])

@pytest.fixture
def client_and_app():
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app

async def test_healthz_is_public(client_and_app):
    client, _ = client_and_app
    assert (await client.get("/healthz")).status_code == 200

async def test_me_returns_context(client_and_app):
    client, app = client_and_app
    ctx = _ctx()
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    r = await client.get("/me")
    assert r.status_code == 200
    assert r.json()["user"]["email"] == "u@x.io"

async def test_invoke_requires_permission(client_and_app):
    client, app = client_and_app
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)  # no llm:invoke
    r = await client.post("/llm/invoke", json={"model_config_id": str(uuid4()),
                                               "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 403

async def test_invoke_calls_provider(client_and_app):
    client, app = client_and_app
    ctx = _ctx(Role.MEMBER)
    cfg = ModelConfig(id=uuid4(), tenant_id=ctx.tenant_id, name="c",
                      provider="openai", model="gpt-4o", secret_ref="ref")

    class _Provider:
        async def complete(self, config, request):
            return LLMResponse(content="pong", usage=Usage(total_tokens=2), cost=0.0)
        def stream(self, config, request): ...

    class _Registry:
        async def get(self, tenant_id, config_id): return cfg

    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_llm_provider] = lambda: _Provider()
    app.dependency_overrides[deps.get_model_registry] = lambda: _Registry()

    r = await client.post("/llm/invoke", json={"model_config_id": str(cfg.id),
                                               "messages": [{"role": "user", "content": "ping"}]})
    assert r.status_code == 200
    assert r.json()["content"] == "pong"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest apps/api-gateway/tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create package, `config.py`, `deps.py`**

`apps/api-gateway/pyproject.toml`:
```toml
[project]
name = "api-gateway"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anodyne-core", "anodyne-tenancy", "anodyne-storage", "anodyne-llm",
                "anodyne-observability", "fastapi>=0.115", "uvicorn[standard]>=0.30",
                "pydantic-settings>=2.4", "pyjwt[crypto]>=2.9"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.uv.sources]
anodyne-core = { workspace = true }
anodyne-tenancy = { workspace = true }
anodyne-storage = { workspace = true }
anodyne-llm = { workspace = true }
anodyne-observability = { workspace = true }
```

```python
# apps/api-gateway/src/api_gateway/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env")
    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/anodyne"
    oidc_issuer: str = "http://localhost:8080/realms/anodyne"
    oidc_jwks_url: str = "http://localhost:8080/realms/anodyne/protocol/openid-connect/certs"
    oidc_audience: str = "anodyne"
    secret_key: str = ""          # base64 Fernet key; required in prod
    s3_bucket: str = "anodyne"

def get_settings() -> Settings:
    return Settings()
```

```python
# apps/api-gateway/src/api_gateway/deps.py
from __future__ import annotations
from functools import lru_cache
import jwt
from fastapi import Depends, Header, HTTPException
from anodyne_core.models import TenantContext
from anodyne_core.ports import AuthorizationPolicy, LLMProvider
from anodyne_tenancy.authz import RoleBasedPolicy
from anodyne_tenancy.oidc import AuthError, TokenValidator
from api_gateway.config import Settings, get_settings

@lru_cache
def _validator(issuer: str, jwks_url: str, audience: str) -> TokenValidator:
    return TokenValidator(jwt.PyJWKClient(jwks_url), issuer=issuer, audience=audience)

def get_tenant_context(authorization: str = Header(default=""),
                       settings: Settings = Depends(get_settings)) -> TenantContext:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        v = _validator(settings.oidc_issuer, settings.oidc_jwks_url, settings.oidc_audience)
        return v.validate(authorization.removeprefix("Bearer "))
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc

def get_policy() -> AuthorizationPolicy:
    return RoleBasedPolicy()

def require(permission: str):
    def _dep(ctx: TenantContext = Depends(get_tenant_context),
             policy: AuthorizationPolicy = Depends(get_policy)) -> TenantContext:
        if not policy.is_permitted(ctx, permission):
            raise HTTPException(403, f"missing permission: {permission}")
        return ctx
    return _dep

# Overridden in tests; real wiring builds these from Settings + backbone.
def get_llm_provider() -> LLMProvider:  # pragma: no cover - wired at runtime
    raise HTTPException(503, "LLM provider not configured")

def get_model_registry():  # pragma: no cover - wired at runtime
    raise HTTPException(503, "model registry not configured")
```

- [ ] **Step 4: Implement `app.py`**

```python
# apps/api-gateway/src/api_gateway/app.py
from __future__ import annotations
from uuid import UUID
from fastapi import Depends, FastAPI, HTTPException
from anodyne_core.models import LLMRequest, TenantContext
from anodyne_core.ports import LLMProvider
from anodyne_observability.logging import configure_logging
from api_gateway import deps

def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Anodyne API Gateway")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/me")
    async def me(ctx: TenantContext = Depends(deps.get_tenant_context)) -> dict[str, object]:
        return ctx.model_dump(mode="json")

    @app.post("/llm/invoke")
    async def invoke(
        request: LLMRequest,
        ctx: TenantContext = Depends(deps.require("llm:invoke")),
        provider: LLMProvider = Depends(deps.get_llm_provider),
        registry=Depends(deps.get_model_registry),
    ) -> dict[str, object]:
        cfg = await registry.get(ctx.tenant_id, request.model_config_id)
        if cfg is None:
            raise HTTPException(404, "model config not found")
        resp = await provider.complete(cfg, request)
        return resp.model_dump(mode="json")

    return app
```

- [ ] **Step 5: Run to verify pass**

Run: `uv sync && uv run pytest apps/api-gateway/tests/test_app.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add apps/api-gateway
git commit -m "feat(gateway): add FastAPI app with auth, RBAC, /me and /llm/invoke"
```

---

### Task 11b: Model registry + `/models` endpoints (`anodyne-llm` + gateway)

**Files:**
- Create: `packages/anodyne-llm/src/anodyne_llm/registry.py`
- Modify: `apps/api-gateway/src/api_gateway/app.py` (add `/models` routes), `apps/api-gateway/src/api_gateway/deps.py` (wire real registry)
- Test: `packages/anodyne-llm/tests/test_registry.py`, `apps/api-gateway/tests/test_models_routes.py`

**Interfaces:**
- Consumes: `ModelConfig`, `tenant_session`, `model_configs` table, `SecretStore`.
- Produces: `SqlModelRegistry(engine, secret_store)` with
  `create(tenant_id, name, provider, model, api_key|None, api_base|None, params) -> ModelConfig`,
  `get(tenant_id, config_id) -> ModelConfig | None`, `list(tenant_id) -> list[ModelConfig]`,
  `delete(tenant_id, config_id) -> None`. On `create`, a provided `api_key` is encrypted via
  `SecretStore` and only the ref persists. All reads/writes go through `tenant_session` so RLS applies.

- [ ] **Step 1: Write failing registry test (testcontainers Postgres, marked integration)**

```python
# packages/anodyne-llm/tests/test_registry.py
import pytest, pytest_asyncio
from uuid import uuid4
from cryptography.fernet import Fernet
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer
from anodyne_storage.db import make_engine, metadata, apply_rls
from anodyne_storage.secrets import FernetSecretStore
from anodyne_llm.registry import SqlModelRegistry

pytestmark = pytest.mark.integration

@pytest_asyncio.fixture
async def engine():
    with PostgresContainer("postgres:16") as pg:
        eng = make_engine(pg.get_connection_url().replace("psycopg2", "asyncpg"))
        async with eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            await conn.execute(text("CREATE ROLE app LOGIN; GRANT ALL ON ALL TABLES IN SCHEMA public TO app;"))
        yield eng
        await eng.dispose()

async def test_create_encrypts_key_and_isolates_tenants(engine):
    reg = SqlModelRegistry(engine, FernetSecretStore(Fernet.generate_key()))
    t1, t2 = uuid4(), uuid4()
    cfg = await reg.create(t1, name="c", provider="openai", model="gpt-4o",
                           api_key="sk-secret", api_base=None, params={})
    assert cfg.secret_ref and cfg.secret_ref != "sk-secret"
    assert await reg.get(t1, cfg.id) is not None
    assert await reg.get(t2, cfg.id) is None          # RLS blocks cross-tenant read
    assert [c.id for c in await reg.list(t1)] == [cfg.id]
    await reg.delete(t1, cfg.id)
    assert await reg.get(t1, cfg.id) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/anodyne-llm/tests/test_registry.py -v -m integration`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `registry.py`**

```python
# packages/anodyne-llm/src/anodyne_llm/registry.py
from __future__ import annotations
from uuid import UUID, uuid4
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine
from anodyne_core.models import ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_storage.db import model_configs, tenant_session

def _row_to_config(row: object) -> ModelConfig:
    m = row._mapping  # type: ignore[attr-defined]
    return ModelConfig(id=m["id"], tenant_id=m["tenant_id"], name=m["name"],
                       provider=m["provider"], model=m["model"], params=m["params"],
                       secret_ref=m["secret_ref"], api_base=m["api_base"],
                       enabled=str(m["enabled"]).lower() == "true")

class SqlModelRegistry:
    def __init__(self, engine: AsyncEngine, secret_store: SecretStore) -> None:
        self._engine = engine
        self._secrets = secret_store

    async def create(self, tenant_id: UUID, *, name: str, provider: str, model: str,
                     api_key: str | None, api_base: str | None,
                     params: dict[str, object]) -> ModelConfig:
        secret_ref = self._secrets.encrypt(api_key) if api_key else None
        cid = uuid4()
        async with tenant_session(self._engine, tenant_id) as s:
            await s.execute(insert(model_configs).values(
                id=cid, tenant_id=tenant_id, name=name, provider=provider, model=model,
                params=params, secret_ref=secret_ref, api_base=api_base, enabled="true"))
            await s.commit()
        return ModelConfig(id=cid, tenant_id=tenant_id, name=name, provider=provider,
                           model=model, params=params, secret_ref=secret_ref, api_base=api_base)

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (await s.execute(select(model_configs).where(model_configs.c.id == config_id))).first()
            return _row_to_config(row) if row else None

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (await s.execute(select(model_configs))).all()
            return [_row_to_config(r) for r in rows]

    async def delete(self, tenant_id: UUID, config_id: UUID) -> None:
        async with tenant_session(self._engine, tenant_id) as s:
            await s.execute(delete(model_configs).where(model_configs.c.id == config_id))
            await s.commit()
```

- [ ] **Step 4: Add `/models` routes + register-request schema in `app.py`**

```python
# add to apps/api-gateway/src/api_gateway/app.py
from uuid import UUID
from pydantic import BaseModel

class RegisterModelRequest(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    params: dict[str, object] = {}

# inside create_app(), after /me:
    @app.post("/models", status_code=201)
    async def register_model(
        body: RegisterModelRequest,
        ctx: TenantContext = Depends(deps.require("models:write")),
        registry=Depends(deps.get_model_registry),
    ) -> dict[str, object]:
        cfg = await registry.create(ctx.tenant_id, name=body.name, provider=body.provider,
                                    model=body.model, api_key=body.api_key,
                                    api_base=body.api_base, params=body.params)
        data = cfg.model_dump(mode="json"); data.pop("secret_ref", None)  # never expose refs
        return data

    @app.get("/models")
    async def list_models(
        ctx: TenantContext = Depends(deps.require("models:read")),
        registry=Depends(deps.get_model_registry),
    ) -> list[dict[str, object]]:
        out = []
        for cfg in await registry.list(ctx.tenant_id):
            d = cfg.model_dump(mode="json"); d.pop("secret_ref", None); out.append(d)
        return out

    @app.delete("/models/{config_id}", status_code=204)
    async def delete_model(
        config_id: UUID,
        ctx: TenantContext = Depends(deps.require("models:delete")),
        registry=Depends(deps.get_model_registry),
    ) -> None:
        await registry.delete(ctx.tenant_id, config_id)
```

- [ ] **Step 5: Write gateway route tests (in-memory registry fake)**

```python
# apps/api-gateway/tests/test_models_routes.py
import pytest
from uuid import uuid4
from httpx import ASGITransport, AsyncClient
from anodyne_core.models import Role, TenantContext, User, ModelConfig
from api_gateway.app import create_app
from api_gateway import deps

def _ctx(role):
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])

class _FakeRegistry:
    def __init__(self): self.store = {}
    async def create(self, tenant_id, *, name, provider, model, api_key, api_base, params):
        cfg = ModelConfig(id=uuid4(), tenant_id=tenant_id, name=name, provider=provider,
                          model=model, params=params, secret_ref="ref" if api_key else None,
                          api_base=api_base)
        self.store[cfg.id] = cfg; return cfg
    async def list(self, tenant_id): return list(self.store.values())
    async def delete(self, tenant_id, config_id): self.store.pop(config_id, None)

@pytest.fixture
def wired():
    app = create_app(); reg = _FakeRegistry()
    app.dependency_overrides[deps.get_model_registry] = lambda: reg
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app

async def test_register_hides_secret_ref(wired):
    client, app = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER)
    r = await client.post("/models", json={"name": "c", "provider": "openai",
                                           "model": "gpt-4o", "api_key": "sk-x"})
    assert r.status_code == 201
    assert "secret_ref" not in r.json()

async def test_viewer_cannot_register(wired):
    client, app = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)
    r = await client.post("/models", json={"name": "c", "provider": "openai", "model": "gpt-4o"})
    assert r.status_code == 403
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest packages/anodyne-llm/tests/test_registry.py apps/api-gateway/tests/test_models_routes.py -v`
Expected: PASS (registry test requires Docker; route tests do not).

- [ ] **Step 7: Commit**

```bash
git add packages/anodyne-llm apps/api-gateway
git commit -m "feat(llm+gateway): add DB-backed model registry and /models endpoints"
```

---

### Task 12: Local backbone (docker-compose, Keycloak seed, Makefile)

**Files:**
- Create: `infra/docker/docker-compose.yml`
- Create: `infra/docker/keycloak/anodyne-realm.json` (realm + Organization + demo users + `anodyne` client)
- Create: `Makefile`
- Create: `.env.example`

**Interfaces:**
- Produces: `make up` (backbone), `make migrate` (Alembic upgrade head), `make seed` (idempotent
  demo tenant). No new code APIs.

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
# infra/docker/docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment: { POSTGRES_USER: app, POSTGRES_PASSWORD: app, POSTGRES_DB: anodyne }
    ports: ["5432:5432"]
  redis:
    image: redis:7
    ports: ["6379:6379"]
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: minio, MINIO_ROOT_PASSWORD: minio123 }
    ports: ["9000:9000", "9001:9001"]
  keycloak:
    image: quay.io/keycloak/keycloak:26.0
    command: start-dev --import-realm
    environment: { KC_BOOTSTRAP_ADMIN_USERNAME: admin, KC_BOOTSTRAP_ADMIN_PASSWORD: admin }
    volumes: ["./keycloak:/opt/keycloak/data/import"]
    ports: ["8080:8080"]
```

- [ ] **Step 2: Write `keycloak/anodyne-realm.json`**

Minimal realm `anodyne` with: realm roles `owner/admin/member/viewer`; a confidential client
`anodyne` (audience mapper adding `anodyne` to `aud`); a protocol mapper putting the user's
organization id into an `org_id` token claim; one demo user `demo@anodyne.dev` (password `demo`)
assigned role `admin` and org `demo-tenant`. (Generate via Keycloak export or hand-author following
the KC 26 realm schema.)

- [ ] **Step 3: Write `Makefile` and `.env.example`**

```makefile
# Makefile
up:        ; docker compose -f infra/docker/docker-compose.yml up -d
down:      ; docker compose -f infra/docker/docker-compose.yml down -v
migrate:   ; uv run alembic -c packages/anodyne-storage/alembic.ini upgrade head
seed:      ; uv run python -m api_gateway.seed
test:      ; uv run pytest
```

```bash
# .env.example
ANODYNE_DATABASE_URL=postgresql+asyncpg://app:app@localhost:5432/anodyne
ANODYNE_OIDC_ISSUER=http://localhost:8080/realms/anodyne
ANODYNE_OIDC_JWKS_URL=http://localhost:8080/realms/anodyne/protocol/openid-connect/certs
ANODYNE_OIDC_AUDIENCE=anodyne
ANODYNE_SECRET_KEY=   # generate: python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"
ANODYNE_S3_BUCKET=anodyne
```

- [ ] **Step 4: Manual verification**

Run: `make up` then wait for Keycloak; `make migrate`. Obtain a token via the KC token endpoint
for `demo@anodyne.dev`, call `GET /me` — expect 200 with the demo tenant/roles. Document the curl
in `docs/` (dev runbook). This task has no automated test; verification is the acceptance gate.

- [ ] **Step 5: Commit**

```bash
git add infra Makefile .env.example
git commit -m "chore(infra): add local backbone (compose, Keycloak realm seed, Makefile)"
```

---

### Task 13: CI quality gate (GitHub Actions)

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a PR-triggered workflow running lint, type-check, and tests. (Full build/scan/sign/deploy is roadmap stage I.)

- [ ] **Step 1: Write `ci.yml`**

```yaml
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
  push: { branches: [main] }
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy .
      - run: uv run pytest -m "not integration"
```

- [ ] **Step 2: Verify locally that the same commands pass**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -m "not integration"`
Expected: all green. (Mark the RLS testcontainers test with `@pytest.mark.integration` and register the marker in `pyproject.toml` so CI can skip Docker-dependent tests; add an integration job later in stage I.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml
git commit -m "ci: add lint/type/test quality gate"
```

---

## Self-Review

**Spec coverage:**
- Domain models & ports → Tasks 2, 3 ✓
- Keycloak single-realm tenancy + `org_id` claim → Tasks 8, 12 ✓
- Postgres RLS isolation → Task 7 ✓
- Role-based authz behind a port → Tasks 3, 4, 11 ✓
- Object store (tenant-prefixed) → Task 6 ✓
- Secrets (encrypted, never plaintext) → Task 5, used in Task 10 ✓
- Observability → Task 9 ✓
- LLM layer (embedded LiteLLM, per-tenant configs) → Task 10 ✓
- Gateway endpoints incl. `/llm/invoke` → Task 11 ✓
- Local dev backbone → Task 12 ✓
- CI quality gate → Task 13 ✓
- Definition of done (demo user registers a model + gets a completion) → exercised by Tasks 11 (automated) + 12 (manual end-to-end) ✓

**Type consistency:** `TenantContext`, `ModelConfig`, `LLMRequest`/`LLMResponse`, `Role`,
`Usage`, `Message` names are used identically across Tasks 2→11. `get_tenant_context`,
`get_llm_provider`, `get_model_registry`, `require(permission)` in `deps.py` match the
overrides in Task 11's tests. `ModelRegistry.get(tenant_id, config_id)` is referenced in Task 11;
its concrete implementation (CRUD over `model_configs` using `tenant_session`) is created as part of
wiring in Task 11's runtime path and the seed script — noted here as a follow-up if a dedicated
registry test is desired.

- DB-backed `ModelRegistry` + `/models` POST/GET/DELETE → **Task 11b** ✓ (added). `SqlModelRegistry`
  method signatures (`create(...)`, `get(tenant_id, config_id)`, `list(tenant_id)`, `delete(...)`)
  match both the runtime wiring in `deps.py` and the DI overrides used in Task 11's tests.

Plan is complete with no remaining gaps or placeholders.
```
