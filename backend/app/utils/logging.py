import structlog
import logging
from typing import Optional

from app.config import settings


def _is_truthy(value: Optional[bool]) -> bool:
    """Safely interpret a possibly None or non‑boolean debug flag."""
    return bool(value)


def configure_logging() -> None:
    """
    Configure the root logger and structlog.

    This function is idempotent: calling it multiple times will not add duplicate
    handlers. It also tolerates missing or None ``settings.debug`` values.
    """
    # Determine log level, treating None as False (i.e., INFO level)
    debug_enabled = _is_truthy(getattr(settings, "debug", False))
    level = logging.DEBUG if debug_enabled else logging.INFO

    # Avoid adding duplicate handlers if configure_logging has already been called
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(message)s")

    # Configure structlog with robust processors list
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose renderer based on debug flag, handling empty or unexpected values
    if debug_enabled:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()