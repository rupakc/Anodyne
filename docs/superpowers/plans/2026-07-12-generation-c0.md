# Generation C0 — Foundation + Tabular-from-Description Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Generation Engine foundation proven by a local, UI-driven vertical slice: describe a tabular dataset → review the LLM-proposed schema → generate (Temporal + Ray) → download Parquet.

**Architecture:** New `anodyne-dataset` (models+ports), `anodyne-generation` (schema proposer + deterministic sampler), `anodyne-compute` (Ray), `anodyne-workflows` (Temporal); `generation-worker` app hosts the Temporal worker dispatching Ray shard tasks; `api-gateway` gains dataset endpoints; `apps/web` (Next.js) provides an OIDC UI. Hexagonal throughout.

**Tech Stack:** Python 3.12 / FastAPI / Pydantic v2 / SQLAlchemy async / Alembic / temporalio / ray / pyarrow / Faker / numpy. Frontend: Next.js + TypeScript + Tailwind + shadcn + Auth.js (Keycloak). Infra: docker-compose (Temporal auto-setup, Ray head, Ollama). Tests: pytest, temporalio testing, testcontainers, Playwright.

## Global Constraints

- Python **3.12+**, uv workspace, `src/` layout; import names underscore, dirs hyphen.
- New Python packages MUST be registered in root `pyproject.toml` (`[dependency-groups] dev` + `[tool.uv.sources]`) and `uv.lock` regenerated, else `uv sync` breaks repo-wide collection.
- `ruff` + `mypy --strict` clean; `pytest -m "not integration and not e2e"` green on every commit.
- Every tenant-scoped table carries `tenant_id` + an RLS policy; the app connects as the non-superuser `anodyne_app` role.
- Docker-dependent tests marked `integration`; browser tests marked `e2e`. Markers registered in `pyproject.toml`.
- Test files use **globally-unique basenames** (prefix with the package, e.g. `test_dataset_models.py`, `test_llm_registry.py`) — mypy rejects duplicate module basenames across the monorepo. Do NOT add `__init__.py` to `tests/` dirs. Pytest runs in `--import-mode=importlib` (set in `pyproject.toml`).
- Generation must be **deterministic given a seed** (same seed+range ⇒ identical rows).
- Secrets never logged/stored in plaintext. Conventional commits; commit per task.
- Frontend: use the **autumn-pastel** design system (soft ambers/terracotta/dusty rose/sage/cream); invoke the `frontend-design` skill before building UI.

---

### Task 1: `anodyne-dataset` — domain models + ports

**Files:**
- Create: `packages/anodyne-dataset/pyproject.toml`, `src/anodyne_dataset/__init__.py`, `models.py`, `ports.py`
- Test: `packages/anodyne-dataset/tests/test_models.py`
- Modify: root `pyproject.toml` (register package)

**Interfaces:**
- Produces: `Modality`, `SemanticType`, `FieldSpec`, `DatasetSpec`, `GenerationJob`, `JobStatus`, `DatasetVersion`, `ShardArtifact`, and ports `DatasetRepository`, `Generator`, `SchemaProposer`. Consumed by Tasks 2–8.

- [ ] **Step 1: Write failing tests**
```python
# packages/anodyne-dataset/tests/test_models.py
from uuid import uuid4
from anodyne_dataset.models import (
    DatasetSpec, FieldSpec, GenerationJob, JobStatus, Modality, SemanticType)

def test_fieldspec_defaults():
    f = FieldSpec(name="age", semantic_type=SemanticType.INTEGER)
    assert f.nullable is False and f.constraints == {}

def test_datasetspec_is_tabular_description():
    spec = DatasetSpec(id=uuid4(), tenant_id=uuid4(), name="d", description="people",
                       modality=Modality.TABULAR, source="description",
                       fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
                       target_rows=100)
    assert spec.status == "draft" and spec.fields[0].name == "age"

def test_job_progress_bounds():
    j = GenerationJob(id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4())
    assert j.status is JobStatus.PENDING and j.progress == 0.0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/anodyne-dataset/tests/test_models.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Create package + `models.py`**
`pyproject.toml` deps: `["anodyne-core", "pydantic>=2.8"]`, hatchling, `[tool.uv.sources] anodyne-core = {workspace=true}`.
```python
# src/anodyne_dataset/models.py
from __future__ import annotations
from datetime import datetime, UTC
from enum import StrEnum
from uuid import UUID
from pydantic import BaseModel, Field

class Modality(StrEnum):
    TABULAR = "tabular"; TEXT = "text"; IMAGE = "image"; AUDIO = "audio"; VIDEO = "video"

class SemanticType(StrEnum):
    INTEGER = "integer"; FLOAT = "float"; BOOLEAN = "boolean"; CATEGORICAL = "categorical"
    DATETIME = "datetime"; NAME = "name"; EMAIL = "email"; ADDRESS = "address"; TEXT = "text"

class FieldSpec(BaseModel):
    name: str
    semantic_type: SemanticType
    nullable: bool = False
    constraints: dict[str, object] = Field(default_factory=dict)   # min/max/choices/regex/...
    distribution: str | None = None                                # e.g. "normal(30,5)"

class DatasetSpec(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    description: str
    modality: Modality
    source: str                                                    # "description" (C0)
    fields: list[FieldSpec]                                        # field/column specs
    target_rows: int
    directives: dict[str, object] = Field(default_factory=dict)
    status: str = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class JobStatus(StrEnum):
    PENDING = "pending"; RUNNING = "running"; AWAITING_REVIEW = "awaiting_review"
    SUCCEEDED = "succeeded"; FAILED = "failed"

class GenerationJob(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    workflow_id: str | None = None

class DatasetVersion(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    artifact_uri: str
    format: str = "parquet"
    row_count: int = 0
    checksum: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class ShardArtifact(BaseModel):
    shard_index: int
    object_key: str
    row_count: int
```

- [ ] **Step 4: `ports.py`**
```python
# src/anodyne_dataset/ports.py
from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID
from anodyne_dataset.models import DatasetSpec, DatasetVersion, FieldSpec, GenerationJob, ShardArtifact

class DatasetRepository(ABC):
    @abstractmethod
    async def create_spec(self, spec: DatasetSpec) -> None: ...
    @abstractmethod
    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None: ...
    @abstractmethod
    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]: ...
    @abstractmethod
    async def update_spec(self, spec: DatasetSpec) -> None: ...
    @abstractmethod
    async def save_job(self, job: GenerationJob) -> None: ...
    @abstractmethod
    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None: ...
    @abstractmethod
    async def add_version(self, version: DatasetVersion) -> None: ...
    @abstractmethod
    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]: ...

class Generator(ABC):
    @abstractmethod
    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> "pyarrow.Table": ...  # type: ignore[name-defined]

class SchemaProposer(ABC):
    @abstractmethod
    async def propose(self, description: str) -> list[FieldSpec]: ...
```

- [ ] **Step 5: Register in root pyproject, `uv sync`, run tests** — `uv sync && uv run pytest packages/anodyne-dataset -q` → PASS. `uv run mypy . && uv run ruff check .` clean.

- [ ] **Step 6: Commit** — `git commit -m "feat(dataset): add generation domain models and ports"`.

---

### Task 2: `anodyne-generation` — deterministic `TabularSampler`

**Files:**
- Create: `packages/anodyne-generation/pyproject.toml`, `src/anodyne_generation/__init__.py`, `sampler.py`
- Test: `packages/anodyne-generation/tests/test_sampler.py`
- Modify: root `pyproject.toml`

**Interfaces:**
- Consumes: `DatasetSpec`, `FieldSpec`, `SemanticType`, `Generator`.
- Produces: `TabularSampler(Generator)` → `generate(spec, start_row, count, seed) -> pyarrow.Table` (deterministic; column per field; honors `constraints`/`distribution`).

- [ ] **Step 1: Write failing tests**
```python
# packages/anodyne-generation/tests/test_sampler.py
from uuid import uuid4
import pyarrow as pa
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_generation.sampler import TabularSampler

def _spec(fields, rows=50):
    return DatasetSpec(id=uuid4(), tenant_id=uuid4(), name="d", description="",
                       modality=Modality.TABULAR, source="description", schema=fields, target_rows=rows)

def test_deterministic_same_seed():
    spec = _spec([FieldSpec(name="age", semantic_type=SemanticType.INTEGER,
                            constraints={"min": 0, "max": 120})])
    t1 = TabularSampler().generate(spec, 0, 50, seed=7)
    t2 = TabularSampler().generate(spec, 0, 50, seed=7)
    assert t1.equals(t2)
    assert t1.num_rows == 50 and t1.column_names == ["age"]

def test_integer_constraints_respected():
    spec = _spec([FieldSpec(name="age", semantic_type=SemanticType.INTEGER,
                            constraints={"min": 18, "max": 21})])
    col = TabularSampler().generate(spec, 0, 200, seed=1).column("age").to_pylist()
    assert all(18 <= v <= 21 for v in col)

def test_categorical_uses_choices():
    spec = _spec([FieldSpec(name="c", semantic_type=SemanticType.CATEGORICAL,
                            constraints={"choices": ["a", "b"]})])
    col = set(TabularSampler().generate(spec, 0, 100, seed=2).column("c").to_pylist())
    assert col <= {"a", "b"}

def test_disjoint_ranges_differ():
    spec = _spec([FieldSpec(name="x", semantic_type=SemanticType.FLOAT)])
    a = TabularSampler().generate(spec, 0, 10, seed=5).column("x").to_pylist()
    b = TabularSampler().generate(spec, 10, 10, seed=5).column("x").to_pylist()
    assert a != b   # different row offset ⇒ different draws
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/anodyne-generation/tests/test_sampler.py -v` → FAIL.

- [ ] **Step 3: Create package + `sampler.py`**
`pyproject.toml` deps: `["anodyne-core", "anodyne-dataset", "pyarrow>=17", "numpy>=2", "faker>=30"]` + workspace sources.
```python
# src/anodyne_generation/sampler.py
from __future__ import annotations
import numpy as np
import pyarrow as pa
from faker import Faker
from anodyne_dataset.models import DatasetSpec, FieldSpec, SemanticType
from anodyne_dataset.ports import Generator

class TabularSampler(Generator):
    """Deterministic, seeded per-field sampler. Row offset feeds the RNG so shards are disjoint."""

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        columns: dict[str, pa.Array] = {}
        for i, field in enumerate(spec.fields):
            # Independent, reproducible stream per (seed, field, shard offset).
            rng = np.random.default_rng([seed, i, start_row])
            fake = Faker(); Faker.seed(seed * 1_000_003 + i * 7919 + start_row)
            columns[field.name] = self._column(field, count, rng, fake)
        return pa.table(columns)

    def _column(self, f: FieldSpec, n: int, rng: np.random.Generator, fake: Faker) -> pa.Array:
        c = f.constraints
        st = f.semantic_type
        if st is SemanticType.INTEGER:
            lo, hi = int(c.get("min", 0)), int(c.get("max", 100))
            return pa.array(rng.integers(lo, hi + 1, n).tolist(), type=pa.int64())
        if st is SemanticType.FLOAT:
            lo, hi = float(c.get("min", 0.0)), float(c.get("max", 1.0))
            return pa.array((rng.random(n) * (hi - lo) + lo).tolist(), type=pa.float64())
        if st is SemanticType.BOOLEAN:
            return pa.array((rng.random(n) < 0.5).tolist(), type=pa.bool_())
        if st is SemanticType.CATEGORICAL:
            choices = list(c.get("choices", ["a", "b", "c"]))
            idx = rng.integers(0, len(choices), n)
            return pa.array([choices[j] for j in idx])
        if st is SemanticType.DATETIME:
            return pa.array([fake.date_time().isoformat() for _ in range(n)])
        if st is SemanticType.NAME:
            return pa.array([fake.name() for _ in range(n)])
        if st is SemanticType.EMAIL:
            return pa.array([fake.email() for _ in range(n)])
        if st is SemanticType.ADDRESS:
            return pa.array([fake.address().replace("\n", ", ") for _ in range(n)])
        return pa.array([fake.text(max_nb_chars=80) for _ in range(n)])   # TEXT default
```

- [ ] **Step 4: Register, `uv sync`, run tests** → PASS; mypy/ruff clean.
- [ ] **Step 5: Commit** — `git commit -m "feat(generation): add deterministic tabular sampler"`.

---

### Task 3: `anodyne-generation` — `LLMSchemaProposer`

**Files:**
- Create: `packages/anodyne-generation/src/anodyne_generation/proposer.py`
- Test: `packages/anodyne-generation/tests/test_proposer.py`

**Interfaces:**
- Consumes: `SchemaProposer`, `FieldSpec`, `SemanticType`, and `anodyne_core.ports.LLMProvider` + `anodyne_core.models.ModelConfig`/`LLMRequest`.
- Produces: `LLMSchemaProposer(provider, model_config)` with `propose(description) -> list[FieldSpec]`; parses a JSON array of `{name, semantic_type, nullable?, constraints?}`; raises `SchemaProposalError` on malformed output.

- [ ] **Step 1: Write failing tests (mock LLMProvider)**
```python
# packages/anodyne-generation/tests/test_proposer.py
import pytest
from uuid import uuid4
from anodyne_core.models import LLMResponse, ModelConfig, Usage
from anodyne_dataset.models import SemanticType
from anodyne_generation.proposer import LLMSchemaProposer, SchemaProposalError

class _Provider:
    def __init__(self, content): self._c = content
    async def complete(self, config, request):
        return LLMResponse(content=self._c, usage=Usage())
    def stream(self, config, request): ...

_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="ollama", model="llama3")

async def test_parses_valid_schema():
    content = '[{"name":"age","semantic_type":"integer","constraints":{"min":0,"max":120}}]'
    fields = await LLMSchemaProposer(_Provider(content), _CFG).propose("people with ages")
    assert fields[0].name == "age" and fields[0].semantic_type is SemanticType.INTEGER

async def test_malformed_raises():
    with pytest.raises(SchemaProposalError):
        await LLMSchemaProposer(_Provider("not json"), _CFG).propose("x")

async def test_extracts_json_from_fenced_block():
    content = "Sure!\n```json\n[{\"name\":\"n\",\"semantic_type\":\"name\"}]\n```"
    fields = await LLMSchemaProposer(_Provider(content), _CFG).propose("names")
    assert fields[0].semantic_type is SemanticType.NAME
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `proposer.py`**
```python
# src/anodyne_generation/proposer.py
from __future__ import annotations
import json, re
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import FieldSpec
from anodyne_dataset.ports import SchemaProposer

class SchemaProposalError(Exception): ...

_SYSTEM = (
    "You design tabular dataset schemas. Given a description, return ONLY a JSON array of "
    'fields: [{"name": str, "semantic_type": one of '
    "[integer,float,boolean,categorical,datetime,name,email,address,text], "
    '"nullable": bool (optional), "constraints": object (optional, e.g. {"min":0,"max":100} '
    'or {"choices":["a","b"]})}]. No prose.'
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

class LLMSchemaProposer(SchemaProposer):
    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def propose(self, description: str) -> list[FieldSpec]:
        req = LLMRequest(model_config_id=self._cfg.id, messages=[
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=description)])
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
```

- [ ] **Step 4: Run tests** → PASS; mypy/ruff clean.
- [ ] **Step 5: Commit** — `git commit -m "feat(generation): add LLM schema proposer"`.

---

### Task 4: `anodyne-storage` — dataset tables, migration, repository

**Files:**
- Modify: `packages/anodyne-storage/src/anodyne_storage/db.py` (add tables + RLS entries)
- Create: migration `0002_datasets.py`; `packages/anodyne-storage/src/anodyne_storage/dataset_repo.py`
- Test: `packages/anodyne-storage/tests/test_dataset_repo.py` (marked integration)
- Modify: `anodyne-storage/pyproject.toml` deps (`anodyne-dataset`), root sources

**Interfaces:**
- Consumes: `DatasetRepository`, dataset models, `tenant_session`.
- Produces: tables `datasets`, `dataset_versions`, `generation_jobs` (RLS on `tenant_id`); `SqlDatasetRepository(engine)` implementing `DatasetRepository`.

- [ ] **Step 1: Write failing integration test (testcontainers, non-superuser role — mirror `test_rls.py`)**
```python
# packages/anodyne-storage/tests/test_dataset_repo.py
import pytest, pytest_asyncio
from uuid import uuid4
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]
from anodyne_storage.db import make_engine, metadata, apply_rls
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType, GenerationJob

pytestmark = pytest.mark.integration

@pytest_asyncio.fixture
async def engine():
    with PostgresContainer("postgres:16") as pg:
        admin = pg.get_connection_url().replace("psycopg2", "asyncpg")
        eng = make_engine(admin)
        async with eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            await conn.execute(text("CREATE ROLE app LOGIN PASSWORD 'app'"))
            await conn.execute(text("GRANT USAGE ON SCHEMA public TO app"))
            await conn.execute(text("GRANT ALL ON ALL TABLES IN SCHEMA public TO app"))
        await eng.dispose()
        app_eng = make_engine(admin.replace(f"//{pg.username}:{pg.password}@", "//app:app@"))
        yield app_eng
        await app_eng.dispose()

def _spec(tid):
    return DatasetSpec(id=uuid4(), tenant_id=tid, name="d", description="x",
                       modality=Modality.TABULAR, source="description",
                       fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)], target_rows=10)

async def test_spec_crud_is_tenant_isolated(engine):
    repo = SqlDatasetRepository(engine)
    t1, t2 = uuid4(), uuid4()
    s = _spec(t1); await repo.create_spec(s)
    assert (await repo.get_spec(t1, s.id)).name == "d"
    assert await repo.get_spec(t2, s.id) is None          # RLS + explicit filter
    assert [x.id for x in await repo.list_specs(t1)] == [s.id]

async def test_job_roundtrip(engine):
    repo = SqlDatasetRepository(engine); t = uuid4(); s = _spec(t)
    await repo.create_spec(s)
    j = GenerationJob(id=uuid4(), tenant_id=t, dataset_id=s.id)
    await repo.save_job(j)
    assert (await repo.get_job(t, j.id)).dataset_id == s.id
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/anodyne-storage/tests/test_dataset_repo.py -v -m integration` → FAIL.

- [ ] **Step 3: Add tables to `db.py`** — append `datasets` (id pk, tenant_id, name, description, modality, source, field_specs JSONB, target_rows, directives JSONB, status), `generation_jobs` (id pk, tenant_id, dataset_id, status, progress float, message, workflow_id), `dataset_versions` (id pk, tenant_id, dataset_id, artifact_uri, format, row_count, checksum) — all with `tenant_id`. Add each to `_TENANT_TABLES` keyed on `tenant_id`.

- [ ] **Step 4: Migration `0002_datasets.py`** — `down_revision = "0001"`; create the three tables via `op.create_table` and enable/force RLS + `CREATE POLICY tenant_isolation ... USING (tenant_id = current_setting('app.tenant_id', true)::uuid)` for each (mirror `0001`).

- [ ] **Step 5: Implement `dataset_repo.py`** — `SqlDatasetRepository(engine)`; every method uses `tenant_session(engine, tenant_id)`; reads add explicit `.where(<table>.c.tenant_id == tenant_id)` (defense-in-depth); `schema`/`directives` stored as JSON (serialize `FieldSpec` via `model_dump`, rebuild via `model_validate`).
```python
# src/anodyne_storage/dataset_repo.py  (representative — get_spec shown; others follow same shape)
from __future__ import annotations
from uuid import UUID
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine
from anodyne_dataset.models import DatasetSpec, DatasetVersion, FieldSpec, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from anodyne_storage.db import datasets, dataset_versions, generation_jobs, tenant_session

def _spec_from_row(m) -> DatasetSpec:  # type: ignore[no-untyped-def]
    return DatasetSpec(id=m["id"], tenant_id=m["tenant_id"], name=m["name"],
        description=m["description"], modality=m["modality"], source=m["source"],
        fields=[FieldSpec.model_validate(f) for f in m["field_specs"]],
        target_rows=m["target_rows"], directives=m["directives"], status=m["status"])

class SqlDatasetRepository(DatasetRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
    async def create_spec(self, spec: DatasetSpec) -> None:
        async with tenant_session(self._engine, spec.tenant_id) as s:
            await s.execute(insert(datasets).values(
                id=spec.id, tenant_id=spec.tenant_id, name=spec.name, description=spec.description,
                modality=str(spec.modality), source=spec.source,
                field_specs=[f.model_dump(mode="json") for f in spec.fields],
                target_rows=spec.target_rows, directives=spec.directives, status=spec.status))
            await s.commit()
    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (await s.execute(select(datasets).where(
                datasets.c.id == dataset_id, datasets.c.tenant_id == tenant_id))).mappings().first()
            return _spec_from_row(row) if row else None
    # list_specs / update_spec / save_job / get_job / add_version / list_versions: same pattern.
```

- [ ] **Step 6: Run integration test (Docker) + full unit suite** → PASS; mypy/ruff clean.
- [ ] **Step 7: Commit** — `git commit -m "feat(storage): add dataset/job/version tables, migration, repository"`.

---

### Task 5: `anodyne-compute` — Ray shard task

**Files:**
- Create: `packages/anodyne-compute/pyproject.toml`, `src/anodyne_compute/__init__.py`, `ray_tasks.py`
- Test: `packages/anodyne-compute/tests/test_ray_tasks.py` (marked integration — uses local Ray)
- Modify: root `pyproject.toml`

**Interfaces:**
- Consumes: `TabularSampler`, `DatasetSpec`.
- Produces: `generate_shard_bytes(spec, start_row, count, seed) -> bytes` (Parquet bytes for a shard) and a Ray-remote wrapper `remote_generate_shard`; `ray_init(address: str | None)` helper.

- [ ] **Step 1: Write failing test (Ray local mode)**
```python
# packages/anodyne-compute/tests/test_ray_tasks.py
import io, pytest, ray, pyarrow.parquet as pq
from uuid import uuid4
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_compute.ray_tasks import generate_shard_bytes, remote_generate_shard

pytestmark = pytest.mark.integration

def _spec():
    return DatasetSpec(id=uuid4(), tenant_id=uuid4(), name="d", description="",
        modality=Modality.TABULAR, source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)], target_rows=20)

def test_generate_shard_bytes_is_parquet():
    data = generate_shard_bytes(_spec(), 0, 20, 3)
    tbl = pq.read_table(io.BytesIO(data))
    assert tbl.num_rows == 20

def test_ray_remote_matches_local():
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        local = generate_shard_bytes(_spec(), 0, 20, 3)
        remote = ray.get(remote_generate_shard.remote(_spec(), 0, 20, 3))
        assert local == remote
    finally:
        ray.shutdown()
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Create package + `ray_tasks.py`**
`pyproject.toml` deps: `["anodyne-dataset", "anodyne-generation", "ray>=2.35", "pyarrow>=17"]` + sources.
```python
# src/anodyne_compute/ray_tasks.py
from __future__ import annotations
import io
import pyarrow.parquet as pq
import ray
from anodyne_dataset.models import DatasetSpec
from anodyne_generation.sampler import TabularSampler

def generate_shard_bytes(spec: DatasetSpec, start_row: int, count: int, seed: int) -> bytes:
    table = TabularSampler().generate(spec, start_row, count, seed)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()

@ray.remote
def remote_generate_shard(spec: DatasetSpec, start_row: int, count: int, seed: int) -> bytes:
    return generate_shard_bytes(spec, start_row, count, seed)

def ray_init(address: str | None) -> None:
    if not ray.is_initialized():
        ray.init(address=address or "auto", ignore_reinit_error=True)
```

- [ ] **Step 4: Register, run integration test (local Ray)** → PASS; mypy/ruff clean.
- [ ] **Step 5: Commit** — `git commit -m "feat(compute): add Ray shard generation task"`.

---

### Task 6: `anodyne-workflows` — Temporal activities + `GenerationWorkflow`

**Files:**
- Create: `packages/anodyne-workflows/pyproject.toml`, `src/anodyne_workflows/__init__.py`, `activities.py`, `workflow.py`
- Test: `packages/anodyne-workflows/tests/test_workflow.py` (time-skipping env, mocked activities)
- Modify: root `pyproject.toml`

**Interfaces:**
- Produces: activities `plan_shards`, `generate_shards`, `assemble_and_upload`, `register_version`, `set_status`; `GenerationWorkflow` with `approve_schema` signal + `wait_condition` gate; input dataclass `GenerationInput(job_id, dataset_id, tenant_id, target_rows, seed)`.
- Consumes: `SqlDatasetRepository`, `anodyne-compute`, `ObjectStore`.

- [ ] **Step 1: Write failing workflow test (mock activities)**
```python
# packages/anodyne-workflows/tests/test_workflow.py
import uuid, pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from anodyne_workflows.workflow import GenerationWorkflow, GenerationInput

pytestmark = pytest.mark.integration   # needs the temporal test server download

async def test_workflow_runs_after_approval():
    calls: list[str] = []
    @activity.defn(name="plan_shards")
    async def plan_shards(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan"); return [[0, 5], [5, 5]]
    @activity.defn(name="generate_shards")
    async def generate_shards(inp: GenerationInput, shards: list[list[int]]) -> list[str]:
        calls.append("gen"); return ["k0", "k1"]
    @activity.defn(name="assemble_and_upload")
    async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
        calls.append("assemble"); return "s3://bucket/artifact.parquet"
    @activity.defn(name="register_version")
    async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
        calls.append("register")
    @activity.defn(name="set_status")
    async def set_status(inp: GenerationInput, status: str, progress: float) -> None: ...

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, task_queue="gen",
                          workflows=[GenerationWorkflow],
                          activities=[plan_shards, generate_shards, assemble_and_upload,
                                      register_version, set_status]):
            inp = GenerationInput(job_id=str(uuid.uuid4()), dataset_id=str(uuid.uuid4()),
                                  tenant_id=str(uuid.uuid4()), target_rows=10, seed=1)
            handle = await env.client.start_workflow(
                GenerationWorkflow.run, inp, id="wf-1", task_queue="gen")
            await handle.signal(GenerationWorkflow.approve_schema)   # HITL gate
            uri = await handle.result()
    assert uri == "s3://bucket/artifact.parquet"
    assert calls == ["plan", "gen", "assemble", "register"]
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: `workflow.py` + `activities.py`**
`pyproject.toml` deps: `["anodyne-dataset","anodyne-storage","anodyne-compute","anodyne-core","temporalio>=1.7"]` + sources.
```python
# src/anodyne_workflows/workflow.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

@dataclass
class GenerationInput:
    job_id: str
    dataset_id: str
    tenant_id: str
    target_rows: int
    seed: int

@workflow.defn
class GenerationWorkflow:
    def __init__(self) -> None:
        self._approved = False

    @workflow.signal
    def approve_schema(self) -> None:
        self._approved = True

    @workflow.run
    async def run(self, inp: GenerationInput) -> str:
        opts = dict(start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(maximum_attempts=3))
        await workflow.execute_activity("set_status", args=[inp, "awaiting_review", 0.0], **opts)
        await workflow.wait_condition(lambda: self._approved)     # HITL gate
        await workflow.execute_activity("set_status", args=[inp, "running", 0.1], **opts)
        shards = await workflow.execute_activity("plan_shards", args=[inp], **opts)
        keys = await workflow.execute_activity("generate_shards", args=[inp, shards], **opts)
        await workflow.execute_activity("set_status", args=[inp, "running", 0.7], **opts)
        uri = await workflow.execute_activity("assemble_and_upload", args=[inp, keys], **opts)
        await workflow.execute_activity("register_version", args=[inp, uri, inp.target_rows], **opts)
        await workflow.execute_activity("set_status", args=[inp, "succeeded", 1.0], **opts)
        return uri
```
`activities.py`: real implementations bound to infra (constructed in the worker, Task 7): `plan_shards` splits `target_rows` into ~N chunks; `generate_shards` calls `anodyne_compute.remote_generate_shard` per shard via Ray and puts each shard's bytes to the object store (`{tenant}/datasets/{dataset}/{job}/shard-{i}.parquet`), returns keys; `assemble_and_upload` concatenates shard tables → one Parquet → object store, returns URI; `register_version` writes a `DatasetVersion` via the repo; `set_status` updates the `GenerationJob` (repo) and publishes progress to Redis. Each activity reads infra from a module-level context object set by the worker.

- [ ] **Step 4: Register, run workflow test** (downloads the Temporal test server on first run — marked integration) → PASS; mypy/ruff clean on non-integration.
- [ ] **Step 5: Commit** — `git commit -m "feat(workflows): add Temporal GenerationWorkflow + activities"`.

---

### Task 7: `apps/generation-worker` — Temporal worker process

**Files:**
- Create: `apps/generation-worker/pyproject.toml`, `src/generation_worker/__init__.py`, `config.py`, `main.py`
- Test: `apps/generation-worker/tests/test_worker_wiring.py`
- Modify: root `pyproject.toml`

**Interfaces:**
- Consumes: `GenerationWorkflow`, activities, `Client.connect`, `Worker`, `SqlDatasetRepository`, `S3ObjectStore`, `ray_init`, settings.
- Produces: `build_worker(client, deps) -> Worker` and a `main()` entrypoint (`python -m generation_worker.main`).

- [ ] **Step 1: Write failing wiring test** — assert `build_worker` registers `GenerationWorkflow` and all five activity names on task queue `"generation"` (inspect the constructed `Worker`), with activities bound to injected fakes (fake repo/object store). No live Temporal needed.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — `config.py` (pydantic-settings: `temporal_address`, `ray_address`, `redis_url`, `database_url`, `s3_*`); `main.py`: build repo/object-store/Redis, `ray_init`, bind activity context, `Client.connect(temporal_address)`, `build_worker`, `await worker.run()`.

- [ ] **Step 4: Register, run test** → PASS; mypy/ruff clean.
- [ ] **Step 5: Commit** — `git commit -m "feat(generation-worker): add Temporal worker process"`.

---

### Task 8: Gateway — dataset endpoints + Temporal client + progress

**Files:**
- Modify: `apps/api-gateway/src/api_gateway/deps.py` (repo, Temporal client, proposer, Redis), `app.py` (routes + WS), `config.py` (temporal/redis)
- Modify: `packages/anodyne-tenancy/src/anodyne_tenancy/authz.py` (add `datasets:read`/`datasets:write`)
- Test: `apps/api-gateway/tests/test_dataset_routes.py`
- Modify: `apps/api-gateway/pyproject.toml` (deps: anodyne-dataset, anodyne-generation, anodyne-workflows, temporalio, redis)

**Interfaces:**
- Produces routes: `POST /datasets`, `GET /datasets`, `GET /datasets/{id}`, `PATCH /datasets/{id}`, `POST /datasets/{id}/generate`, `GET /jobs/{id}`, `WS /jobs/{id}/stream`, `GET /datasets/{id}/versions`, `GET /datasets/{id}/versions/{version_id}/download`.
- DI (overridable): `get_dataset_repo`, `get_schema_proposer`, `get_temporal_client`, `get_redis`.

- [ ] **Step 1: Write failing route tests (fakes via overrides)** — cover: `POST /datasets` returns the proposed schema (fake proposer) and `datasets:write` enforced (viewer→403); `PATCH` updates schema; `POST /generate` starts a workflow (fake Temporal client records `start_workflow`, returns a job) and requires `datasets:write`; `GET /jobs/{id}` returns status; `GET /versions` lists; unauthorized (no token) → 401.
```python
# apps/api-gateway/tests/test_dataset_routes.py  (representative cases)
# - override get_tenant_context (member vs viewer), get_dataset_repo (in-memory fake),
#   get_schema_proposer (returns fixed [FieldSpec]), get_temporal_client (records start_workflow).
# assert POST /datasets 201 + schema; viewer POST 403; POST /generate 202 + job id;
# GET /jobs/{id} 200 status; GET /datasets/{id}/versions 200 list.
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Add permissions** — `_MEMBER` gains `datasets:read` + `datasets:write`; `_VIEWER` gains `datasets:read` (authz.py). Update `test_authz.py` accordingly.

- [ ] **Step 4: Implement deps + routes** — `get_temporal_client` connects to `settings.temporal_address` (overridden in tests); `POST /datasets` builds `DatasetSpec`, calls proposer, persists, returns spec; `POST /datasets/{id}/generate` creates a `GenerationJob`, calls `client.start_workflow(GenerationWorkflow.run, GenerationInput(...), id=f"gen-{job_id}", task_queue="generation")`, stores `workflow_id`, returns 202 + job; `WS /jobs/{id}/stream` subscribes to the Redis channel `job:{id}` and forwards messages; `download` returns a presigned URL from the object store. All routes RBAC-guarded + tenant-scoped.

- [ ] **Step 5: Run tests** → PASS; mypy/ruff clean; full non-integration suite green.
- [ ] **Step 6: Commit** — `git commit -m "feat(gateway): add dataset endpoints, Temporal start, progress stream"`.

---

### Task 9: Infra — Temporal, Ray head, Ollama in compose + Makefile

**Files:**
- Modify: `infra/docker/docker-compose.yml`, `Makefile`, `.env.example`, `docs/dev-runbook.md`

**Interfaces:** produces `make up` (full backbone incl. Temporal/Ray/Ollama) and `make dev` (gateway + generation-worker + web).

- [ ] **Step 1: Add services to `docker-compose.yml`**
```yaml
  temporal:
    image: temporalio/auto-setup:1.25
    depends_on: [postgres]
    environment: { DB: postgres12, DB_PORT: 5432, POSTGRES_USER: postgres,
      POSTGRES_PWD: postgres, POSTGRES_SEEDS: postgres }
    ports: ["7233:7233"]
  temporal-ui:
    image: temporalio/ui:2.31.0
    environment: { TEMPORAL_ADDRESS: temporal:7233 }
    ports: ["8088:8080"]
  ray-head:
    image: rayproject/ray:2.35.0
    command: ray start --head --dashboard-host 0.0.0.0 --block
    ports: ["8265:8265", "6379:6379", "10001:10001"]
    shm_size: "2gb"
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: ["ollama:/root/.ollama"]
volumes: { ollama: {} }
```
(Note: Ray's internal Redis uses 6379; map the app Redis to a different host port to avoid a clash, e.g. `6380:6379`, and set `ANODYNE_REDIS_URL` accordingly.)

- [ ] **Step 2: Makefile + `.env.example`** — add `dev` target running gateway (uvicorn), `python -m generation_worker.main`, and `pnpm --dir apps/web dev` concurrently; add `ANODYNE_TEMPORAL_ADDRESS=localhost:7233`, `ANODYNE_RAY_ADDRESS=ray://localhost:10001`, `ANODYNE_OLLAMA_BASE=http://localhost:11434`, adjusted `ANODYNE_REDIS_URL`. Document `ollama pull llama3` + registering it for the demo tenant in `docs/dev-runbook.md`.

- [ ] **Step 3: Validate** — `docker compose -f infra/docker/docker-compose.yml config` parses; YAML lints. (Full `make up` validated by the user with Docker.)
- [ ] **Step 4: Commit** — `git commit -m "chore(infra): add Temporal, Ray head, and Ollama to local stack"`.

---

### Task 10: `apps/web` — Next.js scaffold + autumn-pastel design system

**Files:**
- Create: `apps/web/` (Next.js App Router, TS, Tailwind, shadcn), `tailwind.config.ts` (autumn-pastel tokens), `app/globals.css`, base layout, `package.json`, `pnpm-workspace.yaml`/`turbo.json` at root
- Test: `apps/web/__tests__/theme.test.ts` (token presence) + typecheck

**Interfaces:** produces the Next.js app shell + theme; consumed by Tasks 11–13.

- [ ] **Step 1:** Invoke the **frontend-design** skill; define the autumn-pastel palette (soft amber `#E8B98A`-ish, terracotta, dusty rose, sage, cream) as CSS variables + Tailwind theme tokens (light + dark). Write a failing test asserting the theme exposes the named tokens.
- [ ] **Step 2:** Scaffold Next.js (`create-next-app` equivalent, App Router, TS, Tailwind), add shadcn, encode tokens, base layout with the palette. `pnpm install`.
- [ ] **Step 3:** `pnpm --dir apps/web lint && pnpm --dir apps/web typecheck && pnpm --dir apps/web build` green; theme test passes.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): scaffold Next.js app with autumn-pastel design system"`.

---

### Task 11: `apps/web` — Keycloak OIDC login (Auth.js)

**Files:** `apps/web/auth.ts`, `app/api/auth/[...nextauth]/route.ts`, `middleware.ts`, `.env.local.example`
**Interfaces:** produces an authenticated session exposing `accessToken` for API calls; protected app routes.

- [ ] **Step 1:** Write a test (or typed unit) asserting the `session` callback surfaces `accessToken` from the `jwt` callback.
- [ ] **Step 2:** Configure Auth.js:
```typescript
// apps/web/auth.ts
import NextAuth from "next-auth"
import Keycloak from "next-auth/providers/keycloak"
export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [Keycloak({
    issuer: process.env.KEYCLOAK_ISSUER,          // http://localhost:8080/realms/anodyne
    clientId: process.env.KEYCLOAK_CLIENT_ID,     // anodyne
    clientSecret: process.env.KEYCLOAK_CLIENT_SECRET,
  })],
  callbacks: {
    jwt({ token, account }) {
      if (account?.access_token) token.accessToken = account.access_token
      return token
    },
    session({ session, token }) {
      (session as any).accessToken = token.accessToken
      return session
    },
  },
})
```
`middleware.ts` uses `auth` to protect `/app/*`; unauthenticated → sign-in.

- [ ] **Step 3:** `pnpm typecheck`/`lint`/`build` green.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): add Keycloak OIDC login via Auth.js"`.

---

### Task 12: `apps/web` — create-from-description wizard

**Files:** `apps/web/lib/api.ts` (typed client using the session access token), `app/app/new/page.tsx` (+ step components)
**Interfaces:** calls `POST /datasets`, `PATCH /datasets/{id}`, `POST /datasets/{id}/generate`.

- [ ] **Step 1:** Component test (React Testing Library) for the wizard state machine: describe → shows proposed schema (mocked api) → edit a field → generate calls the client. 
- [ ] **Step 2:** Implement the typed API client (adds `Authorization: Bearer <accessToken>`), then the 3-step wizard (describe → review/edit schema table → row count + generate), autumn-pastel styled.
- [ ] **Step 3:** `pnpm test`/`typecheck`/`lint`/`build` green.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): add create-from-description wizard"`.

---

### Task 13: `apps/web` — progress view + dataset browser + download

**Files:** `app/app/jobs/[id]/page.tsx` (WS progress), `app/app/datasets/page.tsx` + `[id]/page.tsx` (list/versions/download)
**Interfaces:** WS `/jobs/{id}/stream`, `GET /datasets`, `GET /datasets/{id}/versions`, download link.

- [ ] **Step 1:** Component tests: progress page renders progress from a mocked WS; dataset list renders versions + a download link from mocked api.
- [ ] **Step 2:** Implement the WebSocket progress hook + progress UI, the dataset/version browser, and the download action, autumn-pastel styled.
- [ ] **Step 3:** `pnpm test`/`typecheck`/`lint`/`build` green.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): add job progress, dataset browser, and download"`.

---

### Task 14: End-to-end Playwright happy path (marked e2e)

**Files:** `apps/web/e2e/generate.spec.ts`, `apps/web/playwright.config.ts`
**Interfaces:** drives the full local stack: login → describe → review → generate → download.

- [ ] **Step 1:** Write the Playwright spec: sign in as the demo user, create a dataset from a description, approve the proposed schema, generate, wait for success, download and assert a non-empty `.parquet`.
- [ ] **Step 2:** `playwright.config.ts` targets `http://localhost:3000`; test tagged `@e2e`. Document `make up && make dev && pnpm --dir apps/web exec playwright test` (requires the full local stack + Docker; run by the developer/CI, not in the unit lane).
- [ ] **Step 3: Commit** — `git commit -m "test(web): add Playwright happy-path e2e for generation"`.

---

### Task 15: CI — web pipeline + integration lane

**Files:** Modify `.github/workflows/ci.yml`
**Interfaces:** produces a `web` job (pnpm install/lint/typecheck/build/test) and confirms the Python `integration` job now also covers the new packages' integration tests (dataset repo, ray, workflow).

- [ ] **Step 1:** Add a `web` job (setup-node + pnpm, `pnpm install --frozen-lockfile`, `pnpm --dir apps/web lint typecheck build test`). Keep `quality` (Python) and `integration` jobs; the integration job already runs `-m integration` and will pick up the new markered tests (Temporal test server + Ray + testcontainers are available on ubuntu runners). The Playwright `e2e` lane stays manual/optional (needs the full stack) — document, don't gate C0 on it.
- [ ] **Step 2:** Validate YAML parses; the Python `-m "not integration and not e2e"` lane stays green locally.
- [ ] **Step 3: Commit** — `git commit -m "ci: add web pipeline and cover new integration tests"`.

---

## Self-Review

**Spec coverage:** dataset domain model + ports → T1 ✓; deterministic tabular generation → T2 ✓; LLM schema-from-description → T3 ✓; storage tables + RLS repo → T4 ✓; Ray execution → T5 ✓; Temporal workflow + HITL gate → T6 ✓; worker process → T7 ✓; gateway endpoints (create/review/generate/status/progress/versions/download) + RBAC → T8 ✓; local run (Temporal/Ray/Ollama in compose) → T9 ✓; Web UI (login, wizard, progress, browser, download, autumn-pastel) → T10–T13 ✓; e2e → T14 ✓; CI → T15 ✓. DoD (UI-driven tabular-from-description → download, locally) exercised by T14 + the runbook.

**Placeholders:** none — `activities.py` bodies (T6.3) and repo methods (T4.5) are described with exact object-key shapes and the representative method shown; UI tasks name exact files, routes, and callbacks with grounded Auth.js/Temporal code.

**Type consistency:** `GenerationInput`, `GenerationWorkflow.approve_schema`, activity names (`plan_shards`/`generate_shards`/`assemble_and_upload`/`register_version`/`set_status`) are identical across T6, T7, T8. `DatasetRepository`/`Generator`/`SchemaProposer` signatures match T1 across T2–T8. Permissions `datasets:read`/`datasets:write` defined in T8.3 and used in T8.4.

**Notes for execution:** T6 and T14 integration/e2e tests need Docker (Temporal test server, testcontainers, Ray) — mark them so the unit lane stays clean; validate them where Docker is available. Register every new package in root `pyproject.toml` (recurring Foundation lesson).
