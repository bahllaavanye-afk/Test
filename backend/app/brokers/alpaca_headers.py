"""Shared Alpaca authentication header builder.

All strategies and API handlers that communicate with the Alpaca REST API
should use the :func:`alpaca_headers` function to obtain the required
authentication headers. Centralising header construction ensures a single
source of truth for credential usage and simplifies future updates (e.g.,
adding new header fields)."""

from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Construct the authentication headers required by Alpaca.

    Returns
    -------
    dict[str, str]
        A dictionary containing the ``APCA-API-KEY-ID`` and
        ``APCA-API-SECRET-KEY`` entries populated from the application
        configuration.
    """
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }