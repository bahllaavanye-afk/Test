"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class AlpacaHeaderError(RuntimeError):
    """Raised when building Alpaca authentication headers fails."""


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaHeaderError: If required configuration values are missing or invalid.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key
    except AttributeError as exc:
        logger.error(
            "Alpaca header construction failed: missing configuration attribute",
            extra={"error": "MissingAttribute", "exception": str(exc)},
            exc_info=True,
        )
        raise AlpacaHeaderError("Missing Alpaca API configuration") from exc
    except Exception as exc:
        logger.error(
            "Unexpected error while constructing Alpaca headers",
            extra={"error": "UnexpectedError", "exception": str(exc)},
            exc_info=True,
        )
        raise AlpacaHeaderError("Failed to construct Alpaca headers") from exc

    if not api_key or not secret_key:
        logger.error(
            "Alpaca header construction failed: empty API credentials",
            extra={"error": "EmptyCredentials"},
        )
        raise AlpacaHeaderError("Alpaca API credentials are empty")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }