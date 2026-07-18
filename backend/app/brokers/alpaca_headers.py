"""Shared Alpaca authentication header builder.

Provides a single source of truth for constructing the HTTP headers required
to authenticate requests against the Alpaca REST API. All strategies and
API handlers should import this function rather than constructing the header
dictionary inline, ensuring consistent credentials and simplifying future
changes to authentication mechanisms.
"""

from typing import Dict

from app.config import settings


def alpaca_headers() -> Dict[str, str]:
    """Construct the authentication headers for Alpaca REST API calls.

    Returns:
        Dict[str, str]: A dictionary containing the required ``APCA-API-KEY-ID``
        and ``APCA-API-SECRET-KEY`` header entries populated from the
        application settings.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }