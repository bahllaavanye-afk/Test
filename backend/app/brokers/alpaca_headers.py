"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def _cached_alpaca_headers() -> dict[str, str]:
    """Create the Alpaca authentication headers once and cache them."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The headers are cached after the first call to avoid repeated construction
    overhead. A shallow copy is returned to protect the cached instance from
    accidental mutation by callers.
    """
    return _cached_alpaca_headers().copy()