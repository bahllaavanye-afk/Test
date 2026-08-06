"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The headers are static for the lifetime of the process, so they are
    cached after the first construction to avoid repeated attribute look‑ups.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }