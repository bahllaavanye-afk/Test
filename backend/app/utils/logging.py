import structlog
import logging
from typing import Optional, List, Any
from app.config import settings


def _clamp_logging_level(level: int) -> int:
    """Clamp the logging level to the valid range defined by the logging module."""
    min_level = logging.NOTSET
    max_level = logging.CRITICAL
    return max(min(level, max_level), min_level)


def _build_processors(debug: bool) -> List[Any]:
    """
    Build the list of structlog processors.

    Handles edge cases where the collection might be empty or `debug` is None.
    """
    # Guard against None for the debug flag
    debug = bool(debug)

    base_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose the final renderer based on the debug flag
    renderer = (
        structlog.dev.ConsoleRenderer()
        if debug
        else structlog.processors.JSONRenderer()
    )
    base_processors.append(renderer)

    # Ensure we never return an empty list; fallback to a minimal processor set
    if not base_processors:
        return [structlog.processors.JSONRenderer()]
    return base_processors


def configure_logging(settings_override: Optional[Any] = None) -> None:
    """
    Configure the global logging and structlog settings.

    This function is defensive against None inputs and off‑by‑one errors in level handling.
    An optional ``settings_override`` can be provided for testing; if omitted, the module‑level
    ``settings`` object is used.
    """
    # Resolve the settings source, handling a possible None input
    cfg = settings_override if settings_override is not None else settings

    # Defensive defaults if the provided config lacks expected attributes
    debug_flag = getattr(cfg, "debug", False)
    debug_flag = bool(debug_flag)  # Ensure a boolean, handling None

    # Determine log level with clamping to avoid off‑by‑one level errors
    level = logging.DEBUG if debug_flag else logging.INFO
    level = _clamp_logging_level(level)

    # Prevent duplicate basicConfig calls which can raise a RuntimeError
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")

    processors = _build_processors(debug_flag)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()