from __future__ import annotations

from typing import cast

import structlog


def configure_logging() -> None:
    """Configure structlog for JSON logging with context variables."""
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
    """Get a logger instance by name."""
    return cast(structlog.BoundLogger, structlog.get_logger(name))


def bind_request_context(*, tenant_id: str, request_id: str) -> None:
    """Bind request context (tenant_id, request_id) to contextvars for correlation."""
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, request_id=request_id)
