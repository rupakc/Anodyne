from __future__ import annotations

import io

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import DatasetSpec
from anodyne_llm.adapter import LiteLLMProvider
from anodyne_storage.secrets import FernetSecretStore
from anodyne_text.generator import TextGenerator


def generate_text_shard_bytes(
    spec: DatasetSpec,
    model_config: ModelConfig,
    secret_key: str,
    start_row: int,
    count: int,
    seed: int,
) -> bytes:
    """Generate Parquet-encoded bytes for a text-dataset shard.

    `model_config.secret_ref` (if any) stays encrypted the whole way here --
    only `secret_key` (the raw Fernet key) crosses into this function/the Ray
    wire, and decryption happens lazily inside `LiteLLMProvider` at LLM-call
    time, exactly like the gateway's own request path. No plaintext API key
    is ever constructed or transmitted separately.

    Args:
        spec: Dataset specification (fields define the row shape).
        model_config: The tenant's registered model to generate with.
        secret_key: Raw Fernet key used to decrypt `model_config.secret_ref`.
        start_row: Starting row index for this shard (feeds batch indexing).
        count: Number of rows to generate for this shard.
        seed: Seed passed through to the LLM request and batch indexing.

    Returns:
        Parquet-encoded bytes for the generated shard.
    """
    secret_store = FernetSecretStore(secret_key.encode())
    provider = LiteLLMProvider(secret_store)
    table = TextGenerator(provider, model_config).generate(spec, start_row, count, seed)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@ray.remote
def remote_generate_text_shard(
    spec: DatasetSpec,
    model_config: ModelConfig,
    secret_key: str,
    start_row: int,
    count: int,
    seed: int,
) -> bytes:
    """Ray remote task wrapping `generate_text_shard_bytes`."""
    return generate_text_shard_bytes(spec, model_config, secret_key, start_row, count, seed)
