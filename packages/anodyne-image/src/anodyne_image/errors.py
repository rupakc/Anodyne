from __future__ import annotations


class ImageProviderError(Exception):
    """Raised by an `ImageProvider` adapter (or its factory) on any failure:

    missing configuration (no API key / no GPU pipeline), a non-2xx response
    from an external API, malformed provider output, or an unknown provider
    name. Never raised for network/GPU issues that don't occur in this
    environment (no live calls happen in tests) -- this is the domain error
    boundary adapters translate infra failures into.
    """
