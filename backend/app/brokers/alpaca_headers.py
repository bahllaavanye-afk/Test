"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
import time
from app.config import settings

_logger = logging.getLogger(__name__)


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Logs a structured INFO message containing key metrics:
    - signal_count: placeholder (default 0)
    - execution_time_ms: time taken to build the header dict
    - pnl: placeholder (default 0)
    """
    start = time.perf_counter()
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Structured logging; placeholders can be overridden by upstream context if needed
    _logger.info(
        "alpaca_headers_generated",
        extra={
            "signal_count": 0,
            "execution_time_ms": round(elapsed_ms, 3),
            "pnl": 0,
        },
    )
    return headers