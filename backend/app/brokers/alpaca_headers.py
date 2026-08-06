"""Shared Alpaca authentication header builder.

This module provides a single helper function to construct the HTTP
headers required for authenticating requests to the Alpaca REST API.
All strategies and API handlers should import :func:`alpaca_headers`
instead of manually constructing the header dictionary, ensuring a
single source‑of‑truth for the credential values.

The function reads the API key and secret from the application settings
object, which is populated from environment variables or a configuration
file at runtime.
"""

from __future__ import annotations

from app.config import settings

__all__: list[str] = ["alpaca_headers"]


def alpaca_headers() -> dict[str, str]:
    """Build and return the authentication headers for Alpaca REST API calls.

    Returns
    -------
    dict[str, str]
        A dictionary containing the ``APCA-API-KEY-ID`` and
        ``APCA-API-SECRET-KEY`` entries required by Alpaca.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }