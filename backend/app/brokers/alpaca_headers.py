"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from __future__ import annotations

from typing import Dict

from app.config import settings


def _build_alpaca_header(api_key: str, secret_key: str) -> Dict[str, str]:
    """Construct the authentication header dictionary for Alpaca.

    Args:
        api_key: The Alpaca API key.
        secret_key: The Alpaca secret key.

    Returns:
        A dictionary containing the required authentication headers.
    """
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def alpaca_headers() -> Dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    This function pulls credentials from the application settings and builds
    the appropriate header dictionary using a dedicated helper.
    """
    return _build_alpaca_header(settings.alpaca_api_key, settings.alpaca_secret_key)