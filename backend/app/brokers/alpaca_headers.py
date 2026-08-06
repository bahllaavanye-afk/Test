"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from types import MappingProxyType
from app.config import settings
from typing import Mapping


@lru_cache(maxsize=1)
def _cached_alpaca_headers() -> MappingProxyType:
    """Build and cache the Alpaca authentication header mapping."""
    return MappingProxyType(
        {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
    )


def alpaca_headers() -> Mapping[str, str]:
    """Return an immutable view of Alpaca authentication headers.

    The underlying header mapping is cached for performance; a read‑only
    ``MappingProxyType`` is returned to prevent accidental mutation.
    """
    return _cached_alpaca_headers()