"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from __future__ import annotations

from typing import Dict

from app.config import settings


def alpaca_headers() -> Dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Handles edge cases where configuration values may be missing or empty.
    Raises:
        ValueError: If the API key or secret key is not set.
    """
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)

    if not api_key:
        raise ValueError("Alpaca API key is not configured or is empty.")
    if not secret_key:
        raise ValueError("Alpaca secret key is not configured or is empty.")

    # Return a fresh dictionary each call to avoid accidental mutation.
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }