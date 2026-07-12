"""`anodyne_compute.ray_tasks_text` -- Ray shard generation for text datasets.

The plain byte-generation test mocks `LiteLLMProvider.complete` (no network,
no live Ray) and stays in the fast lane. The Ray-remote-parity test actually
starts a local Ray runtime with real (separate-process) workers, so an
in-process monkeypatch wouldn't reach it -- instead it points litellm at a
tiny local, offline HTTP stub (an OpenAI-chat-completions-shaped server on
127.0.0.1) that both the driver and the Ray worker process can reach over
the loopback interface. No real LLM/network is ever contacted; the test is
marked `integration` because it starts a real local Ray runtime.
"""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Generator as PyGenerator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import ray
from anodyne_compute.ray_tasks_text import generate_text_shard_bytes, remote_generate_text_shard
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_storage.secrets import FernetSecretStore
from cryptography.fernet import Fernet

_CANNED_ROWS = [{"text": f"support row {i}", "label": "a"} for i in range(3)]
_CANNED_CONTENT = json.dumps(_CANNED_ROWS)


class _StubChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-chat-completions-shaped responder: same canned content always."""

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": _CANNED_CONTENT},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence per-request logging in test output


@pytest.fixture
def stub_llm_server() -> PyGenerator[int, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _spec(target_rows: int = 3) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="support tickets",
        modality=Modality.TEXT,
        source="description",
        fields=[
            FieldSpec(name="text", semantic_type=SemanticType.TEXT),
            FieldSpec(name="label", semantic_type=SemanticType.CATEGORICAL),
        ],
        target_rows=target_rows,
    )


def _model_config(port: int, secret_key: str) -> ModelConfig:
    secret_ref = FernetSecretStore(secret_key.encode()).encrypt("sk-test")
    return ModelConfig(
        id=uuid4(),
        tenant_id=uuid4(),
        name="m",
        provider="openai",
        model="gpt-3.5-turbo",
        api_base=f"http://127.0.0.1:{port}",
        secret_ref=secret_ref,
    )


def test_generate_text_shard_bytes_is_parquet(stub_llm_server: int) -> None:
    key = Fernet.generate_key().decode()

    data = generate_text_shard_bytes(_spec(), _model_config(stub_llm_server, key), key, 0, 3, 1)

    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 3
    assert set(table.column_names) == {"text", "label"}


@pytest.mark.integration
def test_ray_remote_matches_local(stub_llm_server: int) -> None:
    key = Fernet.generate_key().decode()
    model_config = _model_config(stub_llm_server, key)
    ray.init(ignore_reinit_error=True)
    try:
        local = generate_text_shard_bytes(_spec(), model_config, key, 0, 3, 1)
        remote = ray.get(remote_generate_text_shard.remote(_spec(), model_config, key, 0, 3, 1))
        assert local == remote
    finally:
        ray.shutdown()
