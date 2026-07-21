import structlog
import logging
from app.config import settings
from typing import List, Callable, Any


def _get_logging_level() -> int:
    """Return the appropriate logging level based on the debug flag."""
    return logging.DEBUG if settings.debug else logging.INFO


def _build_processors() -> List[Callable[[Any], Any]]:
    """Construct the list of structlog processors respecting the debug mode."""
    base_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.debug
        else structlog.processors.JSONRenderer()
    )
    return base_processors + [renderer]


def configure_logging() -> None:
    """Configure the standard logging and structlog settings."""
    level = _get_logging_level()
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=_build_processors(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()