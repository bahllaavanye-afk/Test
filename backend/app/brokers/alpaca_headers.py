"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
from app.config import settings


logger = logging.getLogger(__name__)


class AlpacaHeaderError(RuntimeError):
    """Exception raised when building Alpaca authentication headers fails."""


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        AlpacaHeaderError: If required credentials are missing or invalid.
    """
    try:
        api_key = settings.alpaca_api_key
        secret_key = settings.alpaca_secret_key

        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are missing or empty.")

        return {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    except AttributeError as exc:
        logger.error("Alpaca settings missing required attributes: %s", exc)
        raise AlpacaHeaderError("Missing Alpaca configuration") from exc

    except ValueError as exc:
        logger.error("Invalid Alpaca credentials: %s", exc)
        raise AlpacaHeaderError(str(exc)) from exc

    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error building Alpaca headers")
        raise AlpacaHeaderError("Failed to build Alpaca headers") from exc