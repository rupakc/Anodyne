"""Per-modality `ModalityHandler` implementations + their registration.

Imported once (for its registration side effects) from the bottom of
`anodyne_workflows.activities`. Each handler owns the modality-specific parts
of shard generation and artifact assembly; the shared activities own
everything common. Keeping them here -- rather than editing the shared
activities per modality -- is what makes the dispatch a single lookup site.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_audio.generator import AudioDatasetGenerator
from anodyne_audio.models import AudioManifestItem
from anodyne_compute import remote_generate_shard, remote_generate_text_shard
from anodyne_compute.sample_tasks import remote_generate_shard_from_generator
from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider, ObjectStore
from anodyne_dataset.models import DatasetSpec
from anodyne_graph import engines
from anodyne_graph.from_sample import graphml_to_dataset
from anodyne_graph.models import Edge, GraphDataset, Node, compute_metrics
from anodyne_graph.serialization import from_json_bytes, to_json_bytes
from anodyne_tabular.builder import build_tabular_generator
from anodyne_tabular.io import read_sample
from anodyne_video.generator import VideoDatasetGenerator
from anodyne_video.models import VideoManifest, VideoManifestItem

from anodyne_workflows import image_activities
from anodyne_workflows.modality import register_modality

if TYPE_CHECKING:
    from anodyne_workflows.activities import ActivityContext
    from anodyne_workflows.workflow import GenerationInput


def _shard_key(inp: GenerationInput, index: int) -> str:
    # Tenant-relative: `S3ObjectStore` prepends `{tenant_id}/` itself, so this
    # key must NOT repeat it.
    return f"datasets/{inp.dataset_id}/{inp.job_id}/shard-{index}.parquet"


def _artifact_key(inp: GenerationInput, ext: str) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.{ext}"


def _manifest_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"


# --------------------------------------------------------------------------- #
# Tabular (the default) -- behaviourally identical to the pre-registry C0/C1
# path: `TabularSampler` on Ray for from-description specs, a fitted synthesizer
# for from-sample specs, concatenated into one Parquet artifact.
# --------------------------------------------------------------------------- #
class TabularHandler:
    shard_rows = 50_000
    artifact_format = "parquet"

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        if spec.source == "sample":
            return await self._from_sample(ctx, inp, spec, shards, store)
        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            ref = remote_generate_shard.remote(spec, start, count, inp.seed + i)
            data: bytes = await asyncio.to_thread(ray.get, ref)
            key = _shard_key(inp, i)
            await store.put(key, data)
            keys.append(key)
        return keys

    async def _from_sample(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        """Fit a tabular synthesizer once, then sample each shard on Ray.

        Fitting (not just sampling) happens once per generation job -- refitting
        a statistical/deep model per shard would be wasteful and would break the
        seed-determinism contract (see `anodyne_tabular`'s generators).
        """
        if ctx.profile_repo is None:
            raise RuntimeError(
                "ActivityContext.profile_repo not configured: cannot generate a "
                "source='sample' dataset"
            )
        tenant_id, dataset_id = uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id)
        profile = await ctx.profile_repo.get_profile(tenant_id, dataset_id)
        if profile is None:
            raise ValueError(
                f"dataset {inp.dataset_id} has source='sample' but no profile; "
                "upload a sample before generating"
            )
        sample_bytes = await store.get(profile.sample_uri)
        sample_df = await asyncio.to_thread(read_sample, sample_bytes, profile.sample_filename)
        generator = await asyncio.to_thread(
            build_tabular_generator,
            inp.method,
            profile,
            sample_df,
            epochs=ctx.ctgan_epochs,
            enable_sdv=ctx.enable_sdv,
        )
        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            ref = remote_generate_shard_from_generator.remote(
                generator, spec, start, count, inp.seed + i
            )
            data: bytes = await asyncio.to_thread(ray.get, ref)
            key = _shard_key(inp, i)
            await store.put(key, data)
            keys.append(key)
        return keys

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        tables = []
        for key in keys:
            data = await store.get(key)
            tables.append(pq.read_table(io.BytesIO(data)))
        table = pa.concat_tables(tables) if tables else pa.table({})
        buf = io.BytesIO()
        pq.write_table(table, buf)
        artifact_key = _artifact_key(inp, "parquet")
        await store.put(artifact_key, buf.getvalue())
        return artifact_key


# --------------------------------------------------------------------------- #
# Text -- one LLM call per (batched) row; shards are small; the artifact is
# JSONL plus a sibling manifest.
# --------------------------------------------------------------------------- #
class TextHandler:
    shard_rows = 200
    artifact_format = "jsonl"

    async def _resolve_model_config(
        self, ctx: ActivityContext, inp: GenerationInput
    ) -> ModelConfig:
        if ctx.model_registry is None or inp.model_config_id is None:
            raise ValueError(
                "text generation requires a registered model: no model_registry/"
                "model_config_id configured for this activity context"
            )
        model_config = await ctx.model_registry.get(
            uuid.UUID(inp.tenant_id), uuid.UUID(inp.model_config_id)
        )
        if model_config is None:
            raise ValueError(
                f"model config {inp.model_config_id} not found for tenant {inp.tenant_id}"
            )
        return model_config

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        model_config = await self._resolve_model_config(ctx, inp)
        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            ref = remote_generate_text_shard.remote(
                spec, model_config, ctx.secret_key, start, count, inp.seed + i
            )
            data: bytes = await asyncio.to_thread(ray.get, ref)
            key = _shard_key(inp, i)
            await store.put(key, data)
            keys.append(key)
        return keys

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        """Write the concatenated table as JSONL + a sibling manifest."""
        tables = []
        for key in keys:
            data = await store.get(key)
            tables.append(pq.read_table(io.BytesIO(data)))
        table = pa.concat_tables(tables) if tables else pa.table({})
        rows = table.to_pylist()
        jsonl_bytes = "\n".join(json.dumps(row) for row in rows).encode()
        artifact_key = _artifact_key(inp, "jsonl")
        await store.put(artifact_key, jsonl_bytes)

        manifest = {
            "modality": "text",
            "dataset_id": inp.dataset_id,
            "job_id": inp.job_id,
            "fields": [f.name for f in spec.fields] if spec is not None else [],
            "rows_produced": table.num_rows,
            "model_config_id": inp.model_config_id,
            "seed": inp.seed,
        }
        await store.put(_manifest_key(inp), json.dumps(manifest).encode())
        return artifact_key


# --------------------------------------------------------------------------- #
# Image -- Parquet shards of (item_index/label/prompt/image_bytes/mime_type)
# unpacked into individual image objects + a manifest. Implementation lives in
# `image_activities`; this handler adapts it to the shared interface.
# --------------------------------------------------------------------------- #
class ImageHandler:
    shard_rows = 50_000
    artifact_format = "image_manifest"

    async def _resolve_provider_config(
        self, ctx: ActivityContext, tenant_id: uuid.UUID
    ) -> tuple[ModelConfig, str | None]:
        if ctx.image_registry is None:
            raise RuntimeError(
                "this worker has no image_registry configured; wire one in "
                "generation_worker.main via build_worker/WorkerDeps"
            )
        configs = await ctx.image_registry.list(tenant_id)
        if not configs:
            raise ValueError(
                f"no image provider configured for tenant {tenant_id}; register one via "
                "POST /image-providers first"
            )
        config = configs[0]
        api_key = (
            ctx.secret_store.decrypt(config.secret_ref)
            if config.secret_ref and ctx.secret_store
            else None
        )
        return config, api_key

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        config, api_key = await self._resolve_provider_config(ctx, uuid.UUID(inp.tenant_id))
        return await image_activities.generate_image_shards(
            inp, shards, spec, store, config, api_key
        )

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        return await image_activities.assemble_image_manifest(inp, keys, store)


# --------------------------------------------------------------------------- #
# Audio -- one TTS synthesis per item via the tenant's `AudioProvider`; each
# shard uploads its clips + a manifest fragment, merged into one manifest.
# --------------------------------------------------------------------------- #
def _audio_item_key(inp: GenerationInput, index: int, fmt: str) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/audio/item-{index}.{fmt}"


def _audio_shard_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/audio/manifest-shard-{index}.json"


class AudioHandler:
    shard_rows = 50_000
    artifact_format = "audio_manifest"

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        if ctx.audio_provider_factory is None:
            raise RuntimeError(
                "no audio_provider_factory configured for audio generation; "
                "see ActivityContext.audio_provider_factory"
            )
        provider = await ctx.audio_provider_factory(spec)
        generator = AudioDatasetGenerator(provider)

        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            pairs = await generator.generate(spec, start, count, inp.seed)
            manifest_items: list[AudioManifestItem] = []
            for plan, result in pairs:
                item_key = _audio_item_key(inp, plan.index, result.format)
                await store.put(item_key, result.audio_bytes)
                manifest_items.append(
                    AudioManifestItem(
                        index=plan.index,
                        object_key=item_key,
                        text=plan.request.text,
                        label=plan.label,
                        voice=plan.request.voice,
                        format=result.format,
                        duration_seconds=result.duration_seconds,
                    )
                )
            shard_key = _audio_shard_key(inp, i)
            payload = json.dumps([m.model_dump(mode="json") for m in manifest_items])
            await store.put(shard_key, payload.encode())
            keys.append(shard_key)
        return keys

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        items: list[dict[str, Any]] = []
        for key in keys:
            data = await store.get(key)
            items.extend(json.loads(data.decode()))
        items.sort(key=lambda d: d["index"])
        manifest = {"dataset_id": inp.dataset_id, "job_id": inp.job_id, "items": items}
        artifact_key = _manifest_key(inp)
        await store.put(artifact_key, json.dumps(manifest).encode())
        return artifact_key


# --------------------------------------------------------------------------- #
# Video -- heavy per-clip generation via a `VideoProvider`; a "shard" batches
# only a few items. Each shard uploads its clips + a manifest fragment, merged
# into one `VideoManifest`.
# --------------------------------------------------------------------------- #
def _video_clip_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/videos/item-{index}.mp4"


def _video_shard_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/videos/manifest-shard-{index}.json"


class VideoHandler:
    shard_rows = 4
    artifact_format = "video-manifest"

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        if ctx.video_registry is None:
            raise RuntimeError(
                "no video_registry configured for video generation; "
                "see ActivityContext.video_registry"
            )
        tenant_id = uuid.UUID(inp.tenant_id)
        configs = [c for c in await ctx.video_registry.list(tenant_id) if c.enabled]
        if not configs:
            raise ValueError(f"no enabled video provider configured for tenant {inp.tenant_id}")
        config = configs[0]
        provider = ctx.video_providers.get(config.provider)
        if provider is None:
            raise ValueError(
                f"no VideoProvider adapter registered for provider {config.provider!r}"
            )
        generator = VideoDatasetGenerator()

        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            results = await generator.generate_items(
                spec,
                provider=provider,
                config=config,
                start_index=start,
                count=count,
                seed=inp.seed,
            )
            item_dicts: list[dict[str, Any]] = []
            for item, content in results:
                key = _video_clip_key(inp, item.index)
                await store.put(key, content)
                updated = item.model_copy(update={"object_key": key})
                item_dicts.append(updated.model_dump(mode="json"))
            shard_key = _video_shard_key(inp, i)
            await store.put(shard_key, json.dumps(item_dicts).encode())
            keys.append(shard_key)
        return keys

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        items: list[dict[str, Any]] = []
        for key in keys:
            data = await store.get(key)
            items.extend(json.loads(data.decode()))
        items.sort(key=lambda d: int(d["index"]))
        manifest = VideoManifest(
            tenant_id=uuid.UUID(inp.tenant_id),
            dataset_id=uuid.UUID(inp.dataset_id),
            job_id=uuid.UUID(inp.job_id),
            items=[VideoManifestItem.model_validate(i) for i in items],
        )
        key = _manifest_key(inp)
        await store.put(key, manifest.model_dump_json().encode())
        return key


# --------------------------------------------------------------------------- #
# Graph -- description -> ontology (proposed at create time, stored in
# spec.directives["ontology"]) -> LLM-generated property graph. For GA the whole
# graph is one shard (shard_rows is large); range-partitioning by node index is
# the seam left for wave GB. Each shard serializes to node-link JSON; assemble
# merges shards into one `GraphDataset` and uploads a single `graph_json`
# artifact.
# --------------------------------------------------------------------------- #
def _graph_shard_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/graph-shard-{index}.json"


class GraphHandler:
    # One shard for typical GA sizes; large so `plan_shards` doesn't split the
    # node budget. Wave GB partitions by community/node-range for huge graphs.
    shard_rows = 1_000_000
    artifact_format = "graph_json"

    async def _resolve_model_config(
        self, ctx: ActivityContext, inp: GenerationInput
    ) -> ModelConfig:
        if ctx.model_registry is None or inp.model_config_id is None:
            raise ValueError(
                "graph generation requires a registered model: no model_registry/"
                "model_config_id configured for this activity context"
            )
        model_config = await ctx.model_registry.get(
            uuid.UUID(inp.tenant_id), uuid.UUID(inp.model_config_id)
        )
        if model_config is None:
            raise ValueError(
                f"model config {inp.model_config_id} not found for tenant {inp.tenant_id}"
            )
        return model_config

    def _provider(self, ctx: ActivityContext) -> LLMProvider:
        if ctx.llm_provider is not None:
            return ctx.llm_provider
        # Build a real provider from the worker's secret material, exactly like
        # the text Ray task does (the raw Fernet key never leaves the worker).
        from anodyne_llm.adapter import LiteLLMProvider
        from anodyne_storage.secrets import FernetSecretStore

        secret_store = ctx.secret_store or FernetSecretStore(ctx.secret_key.encode())
        return LiteLLMProvider(secret_store)

    async def _load_sample(self, spec: DatasetSpec, store: ObjectStore) -> GraphDataset:
        """Load + parse the uploaded sample graph for the from-sample engine.

        Read from ``directives["sample_uri"]`` (no new DB table): node-link JSON
        by default, GraphML when ``directives["sample_format"]`` says so.
        """
        uri = spec.directives.get("sample_uri")
        if not isinstance(uri, str):
            raise ValueError(
                "from-sample graph generation requires directives['sample_uri'] "
                "pointing at an uploaded node-link JSON or GraphML sample"
            )
        data = await store.get(uri)
        fmt = str(spec.directives.get("sample_format", "graph_json")).lower()
        if fmt in ("graphml", "xml"):
            return graphml_to_dataset(data)
        return from_json_bytes(data)

    async def generate_shards(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec,
        shards: list[list[int]],
        store: ObjectStore,
    ) -> list[str]:
        # Engine selection is centralised in anodyne_graph.engines; only the
        # LLM-driven engines (default + hybrid) need a provider/model config.
        needs_llm = engines.needs_llm(spec)
        model_config = await self._resolve_model_config(ctx, inp) if needs_llm else None
        provider = self._provider(ctx) if needs_llm else None
        sample = await self._load_sample(spec, store) if engines.is_from_sample(spec) else None
        keys: list[str] = []
        for i, (start, count) in enumerate(shards):
            # Graph generation is synchronous (drives its own LLM calls via
            # asyncio.run internally), so run it off the event loop.
            dataset = await asyncio.to_thread(
                engines.generate_shard,
                spec,
                provider,
                model_config,
                start,
                count,
                inp.seed,
                i,
                sample=sample,
            )
            key = _graph_shard_key(inp, i)
            await store.put(key, to_json_bytes(dataset))
            keys.append(key)
        return keys

    async def assemble(
        self,
        ctx: ActivityContext,
        inp: GenerationInput,
        spec: DatasetSpec | None,
        keys: list[str],
        store: ObjectStore,
    ) -> str:
        """Merge shard graphs into one `GraphDataset` and upload the artifact.

        Nodes are deduped by id across shards; edges are deduped by
        (type, source, target); the ontology is taken from the first shard.
        """
        ontology = None
        nodes: dict[str, Node] = {}
        edges: dict[tuple[str, str, str], Edge] = {}
        for key in keys:
            data = await store.get(key)
            shard = from_json_bytes(data)
            if ontology is None:
                ontology = shard.ontology
            for node in shard.nodes:
                nodes.setdefault(node.id, node)
            for edge in shard.edges:
                edges.setdefault((edge.type, edge.source, edge.target), edge)

        node_list = list(nodes.values())
        edge_list = list(edges.values())
        if ontology is None:
            from anodyne_graph.models import GraphOntology

            ontology = GraphOntology()
        dataset = GraphDataset(
            ontology=ontology,
            nodes=node_list,
            edges=edge_list,
            metrics=compute_metrics(node_list, edge_list),
        )
        artifact_key = _artifact_key(inp, "json")
        await store.put(artifact_key, to_json_bytes(dataset))
        return artifact_key


register_modality("tabular", TabularHandler())
register_modality("text", TextHandler())
register_modality("image", ImageHandler())
register_modality("audio", AudioHandler())
register_modality("video", VideoHandler())
register_modality("graph", GraphHandler())
