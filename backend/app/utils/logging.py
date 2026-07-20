import structlog
import logging
from app.config import settings

# Logging constants
DEFAULT_LOG_FORMAT = "%(message)s"
TIME_FORMAT_ISO = "iso"
DEBUG_LEVEL = logging.DEBUG
INFO_LEVEL = logging.INFO


def configure_logging() -> None:
    level = DEBUG_LEVEL if settings.debug else INFO_LEVEL
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)

    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.debug
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt=TIME_FORMAT_ISO),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()