from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

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
        await asyncio.to_thread(
            self._c.put_object, Bucket=self._bucket, Key=self._key(key), Body=data
        )

    async def get(self, key: str) -> bytes:
        obj = await asyncio.to_thread(self._c.get_object, Bucket=self._bucket, Key=self._key(key))
        return obj["Body"].read()  # type: ignore[no-any-return]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return await asyncio.to_thread(
            self._c.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(key)},
            ExpiresIn=expires,
        )

    async def list(self, prefix: str) -> list[str]:
        resp = await asyncio.to_thread(
            self._c.list_objects_v2, Bucket=self._bucket, Prefix=self._key(prefix)
        )
        return [o["Key"][len(self._prefix) :] for o in resp.get("Contents", [])]
