import json

import pytest
from anodyne_observability.logging import (
    bind_request_context,
    configure_logging,
    get_logger,
)


def test_logs_are_json_with_bound_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    bind_request_context(tenant_id="t-1", request_id="r-9")
    get_logger("test").info("hello", extra_field=42)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(out)
    assert record["event"] == "hello"
    assert record["tenant_id"] == "t-1"
    assert record["request_id"] == "r-9"
    assert record["extra_field"] == 42
