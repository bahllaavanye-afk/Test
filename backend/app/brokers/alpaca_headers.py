"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings

# Pre‑compute the authentication headers once at import time.
# This avoids repeated attribute look‑ups on the settings object for every call.
_ALPACA_HEADERS: dict[str, str] = {
    "APCA-API-KEY-ID": settings.alpaca_api_key,
    "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
}


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The underlying dictionary is cached at module load to minimise overhead.
    A shallow copy is returned to protect callers from mutating the internal
    cached mapping.
    """
    # Returning a copy preserves the original cached dict from accidental edits.
    return _ALPACA_HEADERS.copy()