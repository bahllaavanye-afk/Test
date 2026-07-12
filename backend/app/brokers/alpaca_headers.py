"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings


logger = logging.getLogger(__name__)


class AlpacaHeaderError(RuntimeError):
    """Raised when Alpaca authentication headers cannot be constructed."""


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaHeaderError: If required configuration values are missing or empty.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key
        if not api_key or not secret_key:
            raise AlpacaHeaderError(
                "Alpaca API credentials are missing or empty in configuration."
            )
        return {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
    except AttributeError as exc:
        logger.error(
            "Alpaca configuration missing required attribute: %s",
            exc,
            exc_info=True,
        )
        raise AlpacaHeaderError(
            "Alpaca configuration missing required attribute."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error while building Alpaca headers.", exc_info=True)
        raise exc