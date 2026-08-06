"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def _cached_alpaca_headers() -> dict[str, str]:
    """Build and cache the Alpaca authentication header dict."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The underlying header dict is cached for performance; a shallow copy is
    returned to prevent accidental mutation of the cached object.
    """
    return _cached_alpaca_headers().copy()