# Generation C4 — Audio Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provider-agnostic audio dataset generation (`modality = audio`) wired through the
existing Temporal `GenerationWorkflow` + `generation-worker`, additively — the tabular (C0) path
is untouched. All tests mock `AudioProvider`; no GPU, no live provider keys.

**Architecture:** `AudioProvider` port + `AudioSynthesisRequest`/`Result` models added to
`anodyne_dataset` (additive). New package `anodyne-audio`: `AudioDatasetGenerator` (orchestration)
+ `ElevenLabsAudioProvider` (external API, httpx) + `SelfHostedAudioProvider` (self-hosted OSS TTS,
injected synth callable — no direct Ray/GPU dependency). `anodyne_compute` gets an additive Ray
actor stub (`SelfHostedTTSActor`) for the production self-hosted path. `anodyne_workflows.activities`
gains a `spec.modality`-keyed branch in `generate_shards`/`assemble_and_upload`/`register_version`
(one new optional `ActivityContext` field, default `None` — backward compatible).
`apps/generation-worker` gets an `AudioProviderFactory` that resolves a tenant's `ModelConfig`
(reusing `anodyne-llm`'s registry/secret pattern) into a concrete adapter. `apps/api-gateway` gets
an additive `POST /datasets/audio` route (reuses `datasets:write`/`datasets:read`).

**Tech stack:** Python 3.12 / Pydantic v2 / httpx / (Ray via an injected callable, not a direct
adapter dependency) / temporalio (unchanged activity registration) / pytest / moto / httpx.MockTransport.

## Global constraints (same as C0)

- `uv` workspace, `src/` layout. Register **every** new package in root `pyproject.toml`
  (`[dependency-groups] dev` + `[tool.uv.sources]`); `uv sync` after each new package; regenerate
  `uv.lock`.
- `ruff` + `mypy --strict` clean; `uv run pytest -q -m "not integration and not e2e"` green and
  growing after every task.
- Test files: globally-unique basenames, prefixed `test_audio_*`. No `tests/__init__.py`.
  `--import-mode=importlib` already set at the root.
- Docker/Ray-dependent tests marked `integration`.
- Multi-tenant: no new tables needed (reuses `model_configs`, `datasets`, `generation_jobs`,
  `dataset_versions` — all already tenant-scoped + RLS). Never log/store plaintext secrets.
- Conventional commits; commit per task.

---

### Task 1: `anodyne_dataset` — `AudioProvider` port + models (additive)

**Files:** Modify `packages/anodyne-dataset/src/anodyne_dataset/models.py`, `ports.py`. Modify
`packages/anodyne-dataset/tests/test_dataset_models.py` (add cases; existing file, no new basename
needed).

**Interfaces produced:** `AudioSynthesisRequest(text, voice, language)`,
`AudioSynthesisResult(audio_bytes, format="wav", duration_seconds)`, `AudioProvider.synthesize(req)
-> AudioSynthesisResult` (async ABC method). Consumed by Task 2 (`anodyne-audio`) and Task 4
(activities).

- [ ] **Step 1 — failing tests** in `test_dataset_models.py`:
```python
import pytest
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult
from anodyne_dataset.ports import AudioProvider

def test_audio_synthesis_request_defaults() -> None:
    r = AudioSynthesisRequest(text="hello")
    assert r.voice is None and r.language is None

def test_audio_synthesis_result_defaults_to_wav() -> None:
    res = AudioSynthesisResult(audio_bytes=b"\x00\x01")
    assert res.format == "wav" and res.duration_seconds is None

async def test_audio_provider_is_an_abstract_async_contract() -> None:
    class _Echo(AudioProvider):
        async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
            return AudioSynthesisResult(audio_bytes=request.text.encode())
    out = await _Echo().synthesize(AudioSynthesisRequest(text="hi"))
    assert out.audio_bytes == b"hi"
    with pytest.raises(TypeError):
        AudioProvider()  # type: ignore[abstract]
```
- [ ] **Step 2:** run → FAIL (`ImportError`).
- [ ] **Step 3:** add to `models.py`:
```python
class AudioSynthesisRequest(BaseModel):
    text: str
    voice: str | None = None
    language: str | None = None

class AudioSynthesisResult(BaseModel):
    audio_bytes: bytes
    format: str = "wav"
    duration_seconds: float | None = None
```
  add to `ports.py`:
```python
class AudioProvider(ABC):
    @abstractmethod
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult: ...
```
- [ ] **Step 4:** `uv run pytest packages/anodyne-dataset -q` → PASS; `ruff`/`mypy` clean.
- [ ] **Step 5: Commit** — `feat(dataset): add AudioProvider port and synthesis models`.

---

### Task 2: `anodyne-audio` — package scaffold + `AudioDatasetGenerator`

**Files:** Create `packages/anodyne-audio/pyproject.toml`,
`src/anodyne_audio/__init__.py`, `models.py`, `generator.py`. Test:
`packages/anodyne-audio/tests/test_audio_generator.py`. Modify root `pyproject.toml`.

**Interfaces:** Consumes `AudioProvider`, `AudioSynthesisRequest/Result`, `DatasetSpec`. Produces
`AudioManifestItem`, `AudioManifest`, `AudioItemPlan`, `AudioDatasetGenerator(provider)` with
`plan_items(spec, start_row, count, seed) -> list[AudioItemPlan]` and `generate(spec, start_row,
count, seed) -> list[tuple[AudioItemPlan, AudioSynthesisResult]]`.

- [ ] **Step 1 — failing tests:**
```python
# packages/anodyne-audio/tests/test_audio_generator.py
from uuid import uuid4
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult, DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_dataset.ports import AudioProvider
from anodyne_audio.generator import AudioDatasetGenerator

class _MockProvider(AudioProvider):
    def __init__(self) -> None:
        self.calls: list[AudioSynthesisRequest] = []
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        self.calls.append(request)
        return AudioSynthesisResult(audio_bytes=request.text.encode(), format="wav", duration_seconds=1.0)

def _spec(directives=None, rows=5):
    return DatasetSpec(id=uuid4(), tenant_id=uuid4(), name="d", description="",
        modality=Modality.AUDIO, source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=rows, directives=directives or {})

async def test_uses_explicit_prompts_and_labels() -> None:
    spec = _spec({"audio": {"prompts": ["hi", "bye"], "labels": ["greeting", "farewell"], "voice": "v1"}})
    provider = _MockProvider()
    pairs = await AudioDatasetGenerator(provider).generate(spec, 0, 2, seed=1)
    assert [p.request.text for p, _ in pairs] == ["hi", "bye"]
    assert [p.label for p, _ in pairs] == ["greeting", "farewell"]
    assert all(p.request.voice == "v1" for p, _ in pairs)
    assert [r.audio_bytes for _, r in pairs] == [b"hi", b"bye"]

async def test_falls_back_to_deterministic_text_without_prompts() -> None:
    spec = _spec(rows=3)
    a = await AudioDatasetGenerator(_MockProvider()).generate(spec, 0, 3, seed=7)
    b = await AudioDatasetGenerator(_MockProvider()).generate(spec, 0, 3, seed=7)
    assert [p.request.text for p, _ in a] == [p.request.text for p, _ in b]   # deterministic
    assert all(p.request.text for p, _ in a)                                  # non-empty

async def test_disjoint_shard_ranges_index_correctly() -> None:
    spec = _spec({"audio": {"prompts": [f"t{i}" for i in range(10)]}}, rows=10)
    pairs = await AudioDatasetGenerator(_MockProvider()).generate(spec, 5, 3, seed=1)
    assert [p.index for p, _ in pairs] == [5, 6, 7]
    assert [p.request.text for p, _ in pairs] == ["t5", "t6", "t7"]

async def test_calls_provider_once_per_item() -> None:
    provider = _MockProvider()
    spec = _spec(rows=4)
    await AudioDatasetGenerator(provider).generate(spec, 0, 4, seed=1)
    assert len(provider.calls) == 4
```
- [ ] **Step 2:** run → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3:** create package.
`pyproject.toml`:
```toml
[project]
name = "anodyne-audio"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["anodyne-core", "anodyne-dataset", "httpx>=0.27", "faker>=30"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.uv.sources]
anodyne-core = { workspace = true }
anodyne-dataset = { workspace = true }
```
`src/anodyne_audio/models.py`:
```python
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

class AudioManifestItem(BaseModel):
    index: int
    object_key: str
    text: str
    label: str | None = None
    voice: str | None = None
    format: str = "wav"
    duration_seconds: float | None = None

class AudioManifest(BaseModel):
    dataset_id: UUID
    job_id: UUID
    items: list[AudioManifestItem] = Field(default_factory=list)
```
`src/anodyne_audio/generator.py`:
```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from faker import Faker
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult, DatasetSpec
from anodyne_dataset.ports import AudioProvider

@dataclass
class AudioItemPlan:
    index: int
    request: AudioSynthesisRequest
    label: str | None

def _audio_directives(spec: DatasetSpec) -> dict[str, object]:
    raw = spec.directives.get("audio")
    return raw if isinstance(raw, dict) else {}

class AudioDatasetGenerator:
    """Orchestrates AudioProvider calls for a shard of a `Modality.AUDIO` DatasetSpec.

    Item text comes from `directives["audio"]["prompts"][i]` if provided (list
    index == row index), else a seeded, deterministic Faker sentence -- so
    "generate N audio items" works with zero directives, mirroring
    `TabularSampler`'s TEXT-field fallback.
    """
    def __init__(self, provider: AudioProvider) -> None:
        self._provider = provider

    def plan_items(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> list[AudioItemPlan]:
        d = _audio_directives(spec)
        prompts = d.get("prompts") if isinstance(d.get("prompts"), list) else None
        labels = d.get("labels") if isinstance(d.get("labels"), list) else None
        voice = d.get("voice") if isinstance(d.get("voice"), str) else None
        language = d.get("language") if isinstance(d.get("language"), str) else None
        plans = []
        for i in range(start_row, start_row + count):
            if prompts is not None and i < len(prompts):
                text = str(prompts[i])
            else:
                fake = Faker()
                Faker.seed(seed * 1_000_003 + i)
                text = fake.sentence()
            label = str(labels[i]) if labels is not None and i < len(labels) else None
            plans.append(AudioItemPlan(
                index=i, label=label,
                request=AudioSynthesisRequest(text=text, voice=voice, language=language),
            ))
        return plans

    async def generate(
        self, spec: DatasetSpec, start_row: int, count: int, seed: int
    ) -> list[tuple[AudioItemPlan, AudioSynthesisResult]]:
        plans = self.plan_items(spec, start_row, count, seed)
        results = await asyncio.gather(*(self._provider.synthesize(p.request) for p in plans))
        return list(zip(plans, results, strict=True))
```
- [ ] **Step 4:** register package in root `pyproject.toml` (dev group + sources), `uv sync`, run
  tests → PASS; `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(audio): add anodyne-audio package with AudioDatasetGenerator`.

---

### Task 3: `anodyne-audio` — `ElevenLabsAudioProvider` (external API adapter)

**Files:** Create `src/anodyne_audio/providers/__init__.py`, `providers/elevenlabs.py`. Test:
`tests/test_audio_elevenlabs_provider.py`.

**Interfaces:** `ElevenLabsAudioProvider(api_key, voice_id, model_id="eleven_multilingual_v2",
http_client=None)` implementing `AudioProvider`. Per ElevenLabs docs (context7 `/websites/elevenlabs_io`):
`POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`, JSON body `{text, model_id,
voice_settings?}`, header `xi-api-key`, response body = raw audio bytes (`audio/mpeg`).

- [ ] **Step 1 — failing tests (httpx.MockTransport, no network):**
```python
# packages/anodyne-audio/tests/test_audio_elevenlabs_provider.py
import httpx, pytest
from anodyne_dataset.models import AudioSynthesisRequest
from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider, ElevenLabsError

def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

async def test_posts_expected_url_headers_and_body() -> None:
    captured = {}
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        return httpx.Response(200, content=b"RIFF...audio...", headers={"content-type": "audio/mpeg"})

    provider = ElevenLabsAudioProvider(api_key="sk-test", voice_id="v1", http_client=_client(handler))
    result = await provider.synthesize(AudioSynthesisRequest(text="hello world"))

    assert captured["url"] == "https://api.elevenlabs.io/v1/text-to-speech/v1"
    assert captured["headers"]["xi-api-key"] == "sk-test"
    assert b'"text":"hello world"' in captured["body"]
    assert result.audio_bytes == b"RIFF...audio...."[:-1]  # exact bytes below
    assert result.format == "mp3"

async def test_raises_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid api key"})
    provider = ElevenLabsAudioProvider(api_key="bad", voice_id="v1", http_client=_client(handler))
    with pytest.raises(ElevenLabsError):
        await provider.synthesize(AudioSynthesisRequest(text="x"))

async def test_voice_override_uses_request_voice_not_default() -> None:
    seen = {}
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"abc")
    provider = ElevenLabsAudioProvider(api_key="k", voice_id="default-voice", http_client=_client(handler))
    await provider.synthesize(AudioSynthesisRequest(text="hi", voice="override-voice"))
    assert seen["url"].endswith("/override-voice")
```
  (Fix the exact-bytes assertion to `result.audio_bytes == b"RIFF...audio...."` matching the
  handler's `content=` value precisely when writing the real test file.)
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement:
```python
# src/anodyne_audio/providers/elevenlabs.py
from __future__ import annotations
import httpx
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult
from anodyne_dataset.ports import AudioProvider

class ElevenLabsError(Exception): ...

_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

class ElevenLabsAudioProvider(AudioProvider):
    """External-API adapter for ElevenLabs text-to-speech.

    A request's own `voice` overrides the adapter's default `voice_id` --
    lets one DatasetSpec mix voices via `directives["audio"]["voice"]` per item
    (future) while still having a sane tenant-level default today.
    """
    def __init__(
        self, *, api_key: str, voice_id: str, model_id: str = "eleven_multilingual_v2",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._client = http_client or httpx.AsyncClient()

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        voice_id = request.voice or self._voice_id
        resp = await self._client.post(
            f"{_BASE_URL}/{voice_id}",
            json={"text": request.text, "model_id": self._model_id},
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            raise ElevenLabsError(f"ElevenLabs TTS failed ({resp.status_code}): {resp.text}")
        content_type = resp.headers.get("content-type", "audio/mpeg")
        fmt = "mp3" if "mpeg" in content_type else content_type.split("/")[-1]
        return AudioSynthesisResult(audio_bytes=resp.content, format=fmt)
```
- [ ] **Step 4:** run tests → PASS; `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(audio): add ElevenLabs external-API audio provider`.

---

### Task 4: `anodyne-audio` — `SelfHostedAudioProvider` (self-hosted OSS adapter interface)

**Files:** Create `src/anodyne_audio/providers/selfhosted.py`. Test:
`tests/test_audio_selfhosted_provider.py`.

**Interfaces:** `SelfHostedAudioProvider(synthesize_fn, format="wav", model_name=...)` where
`synthesize_fn: Callable[[str, str | None], Awaitable[bytes]]`. No Ray/GPU import in this module
— production wiring (Task 6) injects a callable that calls a Ray remote GPU actor.

- [ ] **Step 1 — failing tests:**
```python
# packages/anodyne-audio/tests/test_audio_selfhosted_provider.py
from anodyne_dataset.models import AudioSynthesisRequest
from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider

async def test_delegates_to_injected_synthesize_fn() -> None:
    calls = []
    async def fake_synthesize(text: str, voice: str | None) -> bytes:
        calls.append((text, voice))
        return b"pcm-bytes"
    provider = SelfHostedAudioProvider(fake_synthesize, model_name="xtts_v2")
    result = await provider.synthesize(AudioSynthesisRequest(text="hi", voice="narrator"))
    assert result.audio_bytes == b"pcm-bytes"
    assert result.format == "wav"
    assert calls == [("hi", "narrator")]

async def test_default_format_is_overridable() -> None:
    async def fake_synthesize(text: str, voice: str | None) -> bytes:
        return b"x"
    provider = SelfHostedAudioProvider(fake_synthesize, format="pcm16")
    result = await provider.synthesize(AudioSynthesisRequest(text="t"))
    assert result.format == "pcm16"
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement:
```python
# src/anodyne_audio/providers/selfhosted.py
from __future__ import annotations
from collections.abc import Awaitable, Callable
from anodyne_dataset.models import AudioSynthesisRequest, AudioSynthesisResult
from anodyne_dataset.ports import AudioProvider

SynthesizeFn = Callable[[str, str | None], Awaitable[bytes]]

class SelfHostedAudioProvider(AudioProvider):
    """Adapter for self-hosted OSS TTS/audio models (e.g. XTTS, Bark) served on
    a Ray GPU actor. Has no direct Ray/GPU dependency itself: delegates to an
    injected `synthesize_fn`. Production wiring (`apps/generation-worker`)
    supplies one that calls `anodyne_compute.audio_actor.SelfHostedTTSActor`
    (requires a GPU node pool and the model package — not available here;
    unit tests inject a plain async fake).
    """
    def __init__(self, synthesize_fn: SynthesizeFn, *, format: str = "wav", model_name: str = "self-hosted-tts") -> None:
        self._synthesize_fn = synthesize_fn
        self._format = format
        self._model_name = model_name

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        audio_bytes = await self._synthesize_fn(request.text, request.voice)
        return AudioSynthesisResult(audio_bytes=audio_bytes, format=self._format)
```
- [ ] **Step 4:** run tests → PASS; `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(audio): add self-hosted OSS audio provider adapter`.

---

### Task 5: `anodyne_compute` — `SelfHostedTTSActor` Ray actor stub (additive)

**Files:** Create `packages/anodyne-compute/src/anodyne_compute/audio_actor.py`. Test:
`packages/anodyne-compute/tests/test_audio_actor.py` (marked `integration` — local Ray, no GPU, a
stub loader; no real model weights anywhere in this repo).

**Interfaces:** `@ray.remote class SelfHostedTTSActor(load_model=None, model_name="xtts_v2")` with
`.synthesize(text, voice) -> bytes`; lazily calls `load_model()` on first use. Real model loading
(e.g. via `coqui-tts`) is out of scope here — the actor documents the seam.

- [ ] **Step 1 — failing test:**
```python
# packages/anodyne-compute/tests/test_audio_actor.py
import pytest, ray
from anodyne_compute.audio_actor import SelfHostedTTSActor

pytestmark = pytest.mark.integration

class _StubModel:
    def synthesize(self, text: str, voice: str | None) -> bytes:
        return f"{voice}:{text}".encode()

def test_actor_lazily_loads_model_and_synthesizes() -> None:
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        actor = SelfHostedTTSActor.remote(load_model=_StubModel)
        out = ray.get(actor.synthesize.remote("hello", "v1"))
        assert out == b"v1:hello"
    finally:
        ray.shutdown()

def test_actor_raises_without_a_configured_loader() -> None:
    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        actor = SelfHostedTTSActor.remote()
        with pytest.raises(Exception, match="no model loader configured"):
            ray.get(actor.synthesize.remote("hi", None))
    finally:
        ray.shutdown()
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement:
```python
# src/anodyne_compute/audio_actor.py
"""Ray GPU actor wrapper for self-hosted OSS TTS/audio models (e.g. XTTS, Bark).

Real model loading/inference needs a GPU node and the target model package
(e.g. `coqui-tts` for XTTS) -- neither is available in this environment, so
loading is fully deferred to an injected `load_model` callable. Production
wiring constructs this actor with a loader that imports and loads the real
model onto GPU; nothing here imports a heavy ML package eagerly.
"""
from __future__ import annotations
from collections.abc import Callable
import ray

@ray.remote
class SelfHostedTTSActor:
    def __init__(self, load_model: Callable[[], object] | None = None, model_name: str = "xtts_v2") -> None:
        self._model_name = model_name
        self._load_model = load_model
        self._model: object | None = None

    def _ensure_loaded(self) -> object:
        if self._model is None:
            if self._load_model is None:
                raise RuntimeError(
                    "no model loader configured; production wiring must inject one that loads "
                    f"{self._model_name} onto a GPU node (e.g. via coqui-tts's `TTS` API)"
                )
            self._model = self._load_model()
        return self._model

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        model = self._ensure_loaded()
        return model.synthesize(text, voice)  # type: ignore[no-any-return, attr-defined]
```
- [ ] **Step 4:** register `anodyne-compute` audio module needs no new pyproject deps (ray already
  present). Run `uv run pytest packages/anodyne-compute -q -m integration` → PASS locally (Ray
  local mode, no Docker); `ruff`/`mypy --strict` clean on the non-integration lane.
- [ ] **Step 5: Commit** — `feat(compute): add self-hosted TTS Ray actor stub for audio`.

---

### Task 6: `anodyne_workflows.activities` — audio modality dispatch (additive)

**Files:** Modify `packages/anodyne-workflows/src/anodyne_workflows/activities.py` (deps:
`anodyne-audio` added to `pyproject.toml`). Test: new
`packages/anodyne-workflows/tests/test_audio_activities.py`.

**Interfaces:** `ActivityContext` gains `audio_provider_factory: Callable[[DatasetSpec],
Awaitable[AudioProvider]] | None = None`. `generate_shards`/`assemble_and_upload`/`register_version`
branch on `spec.modality is Modality.AUDIO`; tabular path is byte-for-byte unchanged (verified: it
runs whenever `spec is None` too, matching every existing tabular test's fake repo).

- [ ] **Step 1 — failing tests (moto S3, fake repo + fake provider factory):**
```python
# packages/anodyne-workflows/tests/test_audio_activities.py
import io, json, uuid
from typing import Any
import boto3, pytest
from moto import mock_aws
from anodyne_dataset.models import (
    AudioSynthesisRequest, AudioSynthesisResult, DatasetSpec, DatasetVersion, FieldSpec,
    GenerationJob, Modality, SemanticType,
)
from anodyne_dataset.ports import AudioProvider, DatasetRepository
from anodyne_workflows.activities import (
    ActivityContext, assemble_and_upload, configure_activities, generate_shards, register_version,
)
from anodyne_workflows.workflow import GenerationInput

_BUCKET = "test-bucket"

class _MockProvider(AudioProvider):
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        return AudioSynthesisResult(audio_bytes=request.text.encode(), format="wav")

class _FakeRepo(DatasetRepository):
    def __init__(self, spec: DatasetSpec) -> None:
        self._spec = spec
        self.versions: list[DatasetVersion] = []
    async def create_spec(self, spec): ...
    async def get_spec(self, tenant_id, dataset_id): return self._spec
    async def list_specs(self, tenant_id): return []
    async def update_spec(self, spec): ...
    async def save_job(self, job): ...
    async def get_job(self, tenant_id, job_id): return None
    async def add_version(self, version): self.versions.append(version)
    async def list_versions(self, tenant_id, dataset_id): return []

@pytest.fixture
def s3_client():
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c

def _audio_spec(tenant_id, dataset_id, rows=4):
    return DatasetSpec(id=dataset_id, tenant_id=tenant_id, name="d", description="",
        modality=Modality.AUDIO, source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=rows, directives={"audio": {"prompts": [f"t{i}" for i in range(rows)]}})

async def test_generate_shards_uploads_items_and_manifest_fragment(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id)
    repo = _FakeRepo(spec)
    configure_activities(ActivityContext(
        repo=repo, s3_bucket=_BUCKET, s3_client=s3_client,
        audio_provider_factory=lambda s: _mock_factory(),
    ))
    inp = GenerationInput(job_id=str(job_id), dataset_id=str(dataset_id), tenant_id=str(tenant_id), target_rows=4, seed=1)

    keys = await generate_shards(inp, [[0, 4]])

    assert len(keys) == 1
    fragment = json.loads(s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{keys[0]}")["Body"].read())
    assert [item["text"] for item in fragment] == ["t0", "t1", "t2", "t3"]
    for item in fragment:
        stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{item['object_key']}")
        assert stored["Body"].read() == item["text"].encode()

async def test_assemble_and_upload_merges_audio_manifest_fragments(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id, rows=2)
    inp = GenerationInput(job_id=str(job_id), dataset_id=str(dataset_id), tenant_id=str(tenant_id), target_rows=2, seed=1)
    fragment_key = f"datasets/{dataset_id}/{job_id}/audio/manifest-shard-0.json"
    fragment = [{"index": 0, "object_key": "x", "text": "t0", "label": None, "voice": None, "format": "wav", "duration_seconds": None}]
    s3_client.put_object(Bucket=_BUCKET, Key=f"{tenant_id}/{fragment_key}", Body=json.dumps(fragment).encode())
    configure_activities(ActivityContext(repo=_FakeRepo(spec), s3_bucket=_BUCKET, s3_client=s3_client))

    artifact_key = await assemble_and_upload(inp, [fragment_key])

    assert artifact_key == f"datasets/{dataset_id}/{job_id}/manifest.json"
    manifest = json.loads(s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{artifact_key}")["Body"].read())
    assert manifest["items"][0]["text"] == "t0"

async def test_register_version_sets_audio_manifest_format() -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id)
    repo = _FakeRepo(spec)
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    inp = GenerationInput(job_id=str(job_id), dataset_id=str(dataset_id), tenant_id=str(tenant_id), target_rows=4, seed=1)

    await register_version(inp, "datasets/x/manifest.json", rows=4)

    assert repo.versions[0].format == "audio_manifest"

def _mock_factory():
    return _MockProvider()
```
  (`_mock_factory` returns a plain value, not a coroutine — write
  `audio_provider_factory=lambda s: _async_wrap(_MockProvider())` with a tiny
  `async def _async_wrap(p): return p` helper in the real test file, since
  `ActivityContext.audio_provider_factory` is `async`.)
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement in `activities.py`:
  - Add imports: `from anodyne_audio.generator import AudioDatasetGenerator`, `from
    anodyne_audio.models import AudioManifestItem`, `from anodyne_dataset.models import
    DatasetSpec, Modality`, `from anodyne_dataset.ports import AudioProvider`.
  - `ActivityContext` gains `audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]]
    | None = None`.
  - New helpers `_audio_item_key`, `_audio_manifest_shard_key`, `_audio_manifest_key`,
    `_generate_audio_shards(ctx, inp, spec, shards, store)`,
    `_assemble_audio_manifest(inp, keys, store)` per the spec's "Artifact shape" decision.
  - `generate_shards`: after `spec = await ctx.repo.get_spec(...)`, add
    `if spec.modality is Modality.AUDIO: return await _generate_audio_shards(ctx, inp, spec, shards, store)`
    before the existing tabular loop.
  - `assemble_and_upload`: add `spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id),
    uuid.UUID(inp.dataset_id))` at the top; `if spec is not None and spec.modality is Modality.AUDIO:
    return await _assemble_audio_manifest(inp, keys, store)` before the existing concat logic.
  - `register_version`: fetch `spec` the same way; `fmt = "audio_manifest" if spec is not None and
    spec.modality is Modality.AUDIO else "parquet"`; pass `format=fmt` to `DatasetVersion(...)`.
  - Add `anodyne-audio` to `packages/anodyne-workflows/pyproject.toml` deps + `[tool.uv.sources]`.
- [ ] **Step 4:** `uv sync`; run `uv run pytest packages/anodyne-workflows -q` (full file, incl.
  pre-existing `test_activities.py` and `test_workflow.py`) → PASS, nothing regressed;
  `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(workflows): dispatch generate/assemble/register by spec.modality for audio`.

---

### Task 7: `apps/generation-worker` — `AudioProviderFactory` + wiring

**Files:** Create `apps/generation-worker/src/generation_worker/audio.py`. Test:
`apps/generation-worker/tests/test_audio_provider_factory.py`. Modify
`apps/generation-worker/src/generation_worker/main.py` (`WorkerDeps`, `build_worker`, `main`).

**Interfaces:** `AudioProviderFactory(registry, secrets).build(spec) -> AudioProvider` (async).
`WorkerDeps` gains `audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] |
None = None`; `build_worker` threads it into `ActivityContext`. `registered_activities()` /
`TASK_QUEUE` / workflow registration unchanged (verified by the existing
`test_worker_wiring.py`, which must still pass unmodified).

- [ ] **Step 1 — failing tests:**
```python
# apps/generation-worker/tests/test_audio_provider_factory.py
from uuid import uuid4
import pytest
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from generation_worker.audio import AudioProviderFactory

class _FakeSecrets:
    def encrypt(self, plaintext: str) -> str: return f"enc:{plaintext}"
    def decrypt(self, ref: str) -> str: return ref.removeprefix("enc:")

class _FakeRegistry:
    def __init__(self, configs): self._configs = configs
    async def get(self, tenant_id, config_id):
        return next((c for c in self._configs if c.id == config_id), None)
    async def list(self, tenant_id):
        return [c for c in self._configs if c.tenant_id == tenant_id]

def _spec(tenant_id, directives=None):
    return DatasetSpec(id=uuid4(), tenant_id=tenant_id, name="d", description="",
        modality=Modality.AUDIO, source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=1, directives=directives or {})

async def test_builds_elevenlabs_provider_for_elevenlabs_config() -> None:
    tid = uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tid, name="m", provider="elevenlabs", model="voice-1", secret_ref="enc:sk-live")
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())
    provider = await factory.build(_spec(tid))
    from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider
    assert isinstance(provider, ElevenLabsAudioProvider)

async def test_builds_selfhosted_provider_for_other_providers() -> None:
    tid = uuid4()
    cfg = ModelConfig(id=uuid4(), tenant_id=tid, name="m", provider="xtts", model="xtts_v2")
    factory = AudioProviderFactory(_FakeRegistry([cfg]), _FakeSecrets())
    provider = await factory.build(_spec(tid))
    from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider
    assert isinstance(provider, SelfHostedAudioProvider)

async def test_prefers_explicit_model_config_id_from_directives() -> None:
    tid = uuid4()
    wanted = ModelConfig(id=uuid4(), tenant_id=tid, name="wanted", provider="elevenlabs", model="v2")
    other = ModelConfig(id=uuid4(), tenant_id=tid, name="other", provider="xtts", model="x")
    factory = AudioProviderFactory(_FakeRegistry([other, wanted]), _FakeSecrets())
    provider = await factory.build(_spec(tid, {"audio": {"model_config_id": str(wanted.id)}}))
    from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider
    assert isinstance(provider, ElevenLabsAudioProvider)

async def test_raises_when_no_audio_provider_configured() -> None:
    tid = uuid4()
    factory = AudioProviderFactory(_FakeRegistry([]), _FakeSecrets())
    with pytest.raises(RuntimeError, match="no audio provider configured"):
        await factory.build(_spec(tid))
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement `audio.py`:
```python
from __future__ import annotations
import asyncio
from typing import Any, Protocol
from uuid import UUID
from anodyne_audio.providers.elevenlabs import ElevenLabsAudioProvider
from anodyne_audio.providers.selfhosted import SelfHostedAudioProvider
from anodyne_core.models import ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import AudioProvider

_ELEVENLABS = "elevenlabs"

class ModelRegistryLike(Protocol):
    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None: ...
    async def list(self, tenant_id: UUID) -> list[ModelConfig]: ...

def _audio_directives(spec: DatasetSpec) -> dict[str, object]:
    raw = spec.directives.get("audio")
    return raw if isinstance(raw, dict) else {}

class AudioProviderFactory:
    """Resolves a tenant's registered `ModelConfig` into a concrete `AudioProvider`.

    Reuses the exact `anodyne-llm` model-registry + encrypted-secret pattern:
    an audio provider IS a `ModelConfig` row (`provider="elevenlabs"` external,
    anything else self-hosted) -- no separate provider-config storage.
    """
    def __init__(self, registry: ModelRegistryLike, secrets: SecretStore) -> None:
        self._registry = registry
        self._secrets = secrets
        self._actor: Any | None = None

    async def build(self, spec: DatasetSpec) -> AudioProvider:
        cfg = await self._resolve_config(spec)
        if cfg.provider == _ELEVENLABS:
            api_key = self._secrets.decrypt(cfg.secret_ref) if cfg.secret_ref else ""
            return ElevenLabsAudioProvider(api_key=api_key, voice_id=cfg.model)
        return SelfHostedAudioProvider(self._ray_synthesize, model_name=cfg.model)

    async def _resolve_config(self, spec: DatasetSpec) -> ModelConfig:
        directives = _audio_directives(spec)
        config_id = directives.get("model_config_id")
        if config_id:
            cfg = await self._registry.get(spec.tenant_id, UUID(str(config_id)))
            if cfg is not None:
                return cfg
        configs = await self._registry.list(spec.tenant_id)
        if not configs:
            raise RuntimeError(f"no audio provider configured for tenant {spec.tenant_id}")
        return configs[0]

    def _actor_handle(self) -> Any:
        if self._actor is None:
            from anodyne_compute.audio_actor import SelfHostedTTSActor
            self._actor = SelfHostedTTSActor.remote()  # requires a GPU node pool + Ray
        return self._actor

    async def _ray_synthesize(self, text: str, voice: str | None) -> bytes:
        import ray
        ref = self._actor_handle().synthesize.remote(text, voice)
        return await asyncio.to_thread(ray.get, ref)  # type: ignore[no-any-return]
```
  Note `test_builds_selfhosted_provider_for_other_providers` only calls `factory.build(...)`, which
  constructs `SelfHostedAudioProvider(self._ray_synthesize, ...)` **without invoking**
  `_ray_synthesize` or `_actor_handle` — no Ray import/init happens in this test.
  Modify `main.py`:
  - `WorkerDeps` gains `audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] |
    None = None`.
  - `build_worker` passes it through to `ActivityContext(..., audio_provider_factory=deps.audio_provider_factory)`.
  - `main()`: if `settings.secret_key`, build `FernetSecretStore` + `SqlModelRegistry(engine, secret_store)`
    + `AudioProviderFactory(registry, secret_store).build`, pass as `audio_provider_factory` to `WorkerDeps`.
- [ ] **Step 4:** run `apps/generation-worker` tests (incl. pre-existing `test_worker_wiring.py`,
  unmodified expectations) → PASS; `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(generation-worker): wire per-tenant AudioProviderFactory into activities`.

---

### Task 8: `apps/api-gateway` — `POST /datasets/audio`

**Files:** Modify `apps/api-gateway/src/api_gateway/app.py`. Test: new
`apps/api-gateway/tests/test_audio_dataset_routes.py`.

**Interfaces:** `CreateAudioDatasetRequest {name, description="", target_rows, directives:
{prompts?, labels?, voice?, language?, model_config_id?}}` → `201` + the created `DatasetSpec`
(`modality: "audio"`). Gated on `datasets:write` (existing permission — no RBAC change needed).

- [ ] **Step 1 — failing tests (mirrors `test_dataset_routes.py`'s fixtures):**
```python
# apps/api-gateway/tests/test_audio_dataset_routes.py
from uuid import UUID, uuid4
from anodyne_core.models import Role
# reuse the same `wired` fixture / `_ctx` helper / `_FakeDatasetRepository` as
# test_dataset_routes.py (import them, don't redefine, to avoid drift) — or
# duplicate the minimal fixture if cross-file imports are undesirable; the
# real test file imports from test_dataset_routes for a single source of truth.

async def test_create_audio_dataset_returns_audio_modality(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post("/datasets/audio", json={
        "name": "greetings", "description": "TTS greetings", "target_rows": 3,
        "directives": {"prompts": ["hi", "hello", "hey"], "voice": "narrator"},
    })

    assert r.status_code == 201
    body = r.json()
    assert body["modality"] == "audio"
    assert body["fields"][0]["name"] == "transcript"
    assert body["directives"]["audio"]["prompts"] == ["hi", "hello", "hey"]
    assert UUID(body["id"]) in repo.specs

async def test_viewer_cannot_create_audio_dataset(wired):  # type: ignore[no-untyped-def]
    client, app, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())
    r = await client.post("/datasets/audio", json={"name": "d", "target_rows": 1})
    assert r.status_code == 403

async def test_audio_dataset_can_then_generate(wired):  # type: ignore[no-untyped-def]
    # proves the existing, unchanged /generate route works for modality=audio.
    client, app, repo, fake_client = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post("/datasets/audio", json={"name": "d", "target_rows": 2})
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 1})

    assert r.status_code == 202
    assert len(fake_client.calls) == 1
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement in `app.py`:
```python
class AudioDirectives(BaseModel):
    prompts: list[str] | None = None
    labels: list[str] | None = None
    voice: str | None = None
    language: str | None = None
    model_config_id: UUID | None = None

class CreateAudioDatasetRequest(BaseModel):
    name: str
    description: str = ""
    target_rows: int
    directives: AudioDirectives = AudioDirectives()

@app.post("/datasets/audio", status_code=201)
async def create_audio_dataset(
    body: CreateAudioDatasetRequest,
    ctx: TenantContext = Depends(deps.require("datasets:write")),
    repo: DatasetRepository = Depends(deps.get_dataset_repo),
) -> dict[str, object]:
    directives = body.directives.model_dump(mode="json", exclude_none=True)
    spec = DatasetSpec(
        id=uuid4(), tenant_id=ctx.tenant_id, name=body.name, description=body.description,
        modality=Modality.AUDIO, source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=body.target_rows, directives={"audio": directives},
    )
    await repo.create_spec(spec)
    return spec.model_dump(mode="json")
```
  (`SemanticType` needs importing in `app.py` alongside the existing `FieldSpec` import.)
- [ ] **Step 4:** add `anodyne-audio` is **not** a gateway dependency (the route builds a plain
  `DatasetSpec`; provider wiring lives only in the worker) — no new gateway pyproject dep needed.
  Run `apps/api-gateway` tests (incl. pre-existing) → PASS; `ruff`/`mypy --strict` clean.
- [ ] **Step 5: Commit** — `feat(gateway): add POST /datasets/audio for audio dataset creation`.

---

### Task 9: Root registration, full-suite verification, self-review

**Files:** `pyproject.toml` (add `anodyne-audio` to dev group + sources — verify Tasks 2–8 already
did this incrementally; this task is the final check + `uv.lock` regeneration).

- [ ] **Step 1:** confirm root `pyproject.toml` lists `anodyne-audio` in both `[dependency-groups]
  dev` and `[tool.uv.sources]`.
- [ ] **Step 2:** `uv sync` (regenerates `uv.lock`); `uv run pytest -q -m "not integration and not
  e2e"` → green, test count higher than the C0 baseline; `uv run pytest -q -m integration` (Ray
  local mode + moto, no Docker required for the audio-specific integration tests) → green where
  runnable.
- [ ] **Step 3:** `uv run ruff check . && uv run ruff format --check .` and `uv run mypy .` → clean.
- [ ] **Step 4: Self-review** against this plan + the spec's Definition of Done; fix any gaps.
- [ ] **Step 5: Commit** (if `uv.lock`/pyproject changed since the last commit) —
  `chore: register anodyne-audio in the workspace and regenerate uv.lock`.
- [ ] **Step 6:** push branch `feat/generation-c4-audio` (no merge).

---

## Self-Review

**Spec coverage:** `AudioProvider` port + models → T1 ✓; orchestration + deterministic planning →
T2 ✓; external-API adapter (ElevenLabs, context7-grounded) → T3 ✓; self-hosted adapter interface →
T4 ✓; Ray/GPU actor stub → T5 ✓; modality dispatch through the Temporal activities (the
"`Generator` selected by `spec.modality`" requirement) → T6 ✓; per-tenant provider config reusing
the `anodyne-llm` pattern, wired into the worker → T7 ✓; gateway route + RBAC → T8 ✓; registration
+ full verification → T9 ✓. CRITICAL constraint (mocked `AudioProvider`, no GPU/network in tests)
holds in every task — verified per-task above.

**Backward compatibility:** T6 and T7 add one new optional field each
(`ActivityContext.audio_provider_factory`, `WorkerDeps.audio_provider_factory`), both defaulting to
`None`; every existing C0 test's fakes are unaffected (traced explicitly: `_FakeDatasetRepository.get_spec`
returns `None` in `test_activities.py`, which routes into the pre-existing tabular branch since
`spec is not None and spec.modality is Modality.AUDIO` is `False`).

**Notes for execution:** Task 5's Ray test needs local Ray (already exercised by C0's
`test_ray_tasks.py`/`test_ray_init.py` in this environment) — no Docker, no GPU. Register
`anodyne-audio` in root `pyproject.toml` the moment it's created (Task 2), not deferred to Task 9,
to keep `uv sync` working for every subsequent task.
