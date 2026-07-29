"""Utility for constructing Alpaca API authentication headers.

This module provides a single helper function that builds the HTTP headers
required for authenticating requests to the Alpaca REST API. All strategies
and API client code should import and use :func:`alpaca_headers` instead of
duplicating the header construction logic, ensuring a single source of truth
for credential handling.

The function pulls the API key and secret from the application's configuration
object, which is populated from environment variables or a secrets manager at
runtime. No parameters are required, and the returned dictionary can be passed
directly to ``requests`` or any HTTP client that accepts a ``headers`` mapping.
"""

from typing import Dict

from app.config import settings


def alpaca_headers() -> Dict[str, str]:
    """Build and return authentication headers for Alpaca REST API calls.

    Returns
    -------
    Dict[str, str]
        A dictionary containing ``APCA-API-KEY-ID`` and ``APCA-API-SECRET-KEY``
        entries populated from the configured Alpaca credentials.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }