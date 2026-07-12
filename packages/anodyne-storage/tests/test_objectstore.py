import boto3
import pytest
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
