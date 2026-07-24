"""Shared Alpaca authentication header builder.

This module provides a single source of truth for constructing the HTTP headers
required to authenticate requests to the Alpaca REST API. All strategies and
API handlers should import :func:`alpaca_headers` instead of recreating the
dictionary inline, ensuring consistency and simplifying future credential
updates.

The function reads the API key and secret from the application settings
(`app.config.settings`) and returns a dictionary suitable for passing to
``requests`` or any HTTP client that accepts a ``headers`` mapping.
"""

from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Build the authentication header dictionary for Alpaca API calls.

    Returns
    -------
    dict[str, str]
        A mapping containing the ``APCA-API-KEY-ID`` and ``APCA-API-SECRET-KEY``
        entries required by Alpaca for request authentication.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }