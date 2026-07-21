"""Shared Alpaca authentication header builder.

Provides a single source of truth for constructing the HTTP headers required
to authenticate requests against the Alpaca REST API. All strategy modules
and API wrappers should import and use :func:`alpaca_headers` instead of
building the header dictionary manually. This helps keep credentials in a
single location and makes it easier to audit and modify authentication
behaviour.

The function reads the API key and secret from the application settings
object defined in ``app.config.settings``.
"""

from typing import Dict
from app.config import settings

__all__ = ["alpaca_headers"]


def alpaca_headers() -> Dict[str, str]:
    """Return the HTTP headers required for Alpaca API authentication.

    The headers include the API key ID and secret key as expected by the
    Alpaca service. The values are sourced from the global ``settings``
    configuration.

    Returns:
        Dict[str, str]: A dictionary with ``APCA-API-KEY-ID`` and
        ``APCA-API-SECRET-KEY`` entries.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }