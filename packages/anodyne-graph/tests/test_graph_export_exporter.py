from __future__ import annotations

from uuid import uuid4

import pytest
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion
from anodyne_graph.errors import UnsupportedGraphExportFormatError
from anodyne_graph.export import GRAPH_SUPPORTED_FORMATS, GraphExporter
from anodyne_graph.models import Edge, GraphDataset, GraphOntology, Node, compute_metrics
from anodyne_graph.serialization import to_json_bytes


class _FakeObjectStore(ObjectStore):
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.puts: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.puts[key] = data

    async def get(self, key: str) -> bytes:
        return self.data

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


def _dataset() -> GraphDataset:
    nodes = [Node(id="p1", type="Person"), Node(id="c1", type="Company")]
    edges = [Edge(id="e1", type="WORKS_AT", source="p1", target="c1")]
    return GraphDataset(
        ontology=GraphOntology(), nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def _version(tenant_id, dataset_id, version_id) -> DatasetVersion:  # type: ignore[no-untyped-def]
    return DatasetVersion(
        id=version_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        artifact_uri="datasets/x/y/graph.json",
        format="graph_json",
        row_count=3,
    )


async def test_export_defaults_to_graph_json_passthrough() -> None:
    tid, did, vid = uuid4(), uuid4(), uuid4()
    store = _FakeObjectStore(to_json_bytes(_dataset()))
    artifact = await GraphExporter().export(_version(tid, did, vid), store)
    assert artifact.format == "graph_json"
    assert artifact.object_key == f"datasets/{did}/{vid}/export.json"
    assert store.puts[artifact.object_key] == to_json_bytes(_dataset())


@pytest.mark.parametrize("fmt", sorted(GRAPH_SUPPORTED_FORMATS))
async def test_export_writes_every_supported_format(fmt: str) -> None:
    tid, did, vid = uuid4(), uuid4(), uuid4()
    store = _FakeObjectStore(to_json_bytes(_dataset()))
    artifact = await GraphExporter().export(_version(tid, did, vid), store, format=fmt)
    assert artifact.format == fmt
    assert artifact.object_key in store.puts
    assert len(store.puts[artifact.object_key]) > 0
    assert artifact.row_count == 3  # 2 nodes + 1 edge


async def test_export_rejects_unsupported_format() -> None:
    tid, did, vid = uuid4(), uuid4(), uuid4()
    store = _FakeObjectStore(to_json_bytes(_dataset()))
    with pytest.raises(UnsupportedGraphExportFormatError):
        await GraphExporter().export(_version(tid, did, vid), store, format="xml-schema")


async def test_export_row_count_is_node_plus_edge_count() -> None:
    tid, did, vid = uuid4(), uuid4(), uuid4()
    store = _FakeObjectStore(to_json_bytes(_dataset()))
    artifact = await GraphExporter().export(_version(tid, did, vid), store, format="ttl")
    assert artifact.row_count == 2 + 1
