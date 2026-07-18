"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        ValueError: If the API key or secret is missing or not a non‑empty string.
    """
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)

    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Alpaca API key is missing or not a non‑empty string.")
    if not isinstance(secret_key, str) or not secret_key:
        raise ValueError("Alpaca secret key is missing or not a non‑empty string.")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }