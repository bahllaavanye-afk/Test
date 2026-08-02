"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
import logging
import time
import contextvars

from app.config import settings

# Context variables that can be set by calling code to provide metrics.
signal_count_var = contextvars.ContextVar("signal_count", default=None)
pnl_var = contextvars.ContextVar("pnl", default=None)

_logger = logging.getLogger(__name__)


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    The function logs execution metrics at INFO level using structured
    logging. Callers can set ``signal_count`` and ``pnl`` via the
    ``signal_count_var`` and ``pnl_var`` context variables to enrich the log.
    """
    start_time = time.perf_counter()

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _logger.info(
        "Alpaca headers generated",
        extra={
            "signal_count": signal_count_var.get(),
            "execution_time_ms": round(elapsed_ms, 3),
            "pnl": pnl_var.get(),
        },
    )
    return headers