"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        ValueError: If the Alpaca API key or secret key is missing or not a
            non‑empty string.
    """
    api_key = settings.alpaca_api_key
    secret_key = settings.alpaca_secret_key

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Alpaca API key must be a non-empty string.")
    if not isinstance(secret_key, str) or not secret_key.strip():
        raise ValueError("Alpaca secret key must be a non-empty string.")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }