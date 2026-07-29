import structlog
import logging
from pydantic import BaseModel, Field, validator
from app.config import settings


class LoggingConfig(BaseModel):
    """
    Schema for logging configuration.

    Attributes
    ----------
    level: str
        The logging level name. Must be one of the standard logging levels.
    format: str
        The log message format string passed to ``logging.basicConfig``.
    json: bool
        Determines whether logs are rendered as JSON (production) or as a human‑readable console
        output (debug).
    """

    level: str = Field(
        ...,
        description="Logging level name (e.g., 'INFO', 'DEBUG').",
        example="INFO",
    )
    format: str = Field(
        "%(message)s",
        description="Format string for log messages.",
        example="%(asctime)s - %(levelname)s - %(message)s",
    )
    json: bool = Field(
        False,
        description="Render logs as JSON when True; otherwise use console renderer.",
        example=False,
    )

    @validator("level")
    def validate_level(cls, v: str) -> str:
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        lvl = v.upper()
        if lvl not in allowed_levels:
            raise ValueError(f"Invalid log level: {v!r}. Must be one of {sorted(allowed_levels)}.")
        return lvl


def configure_logging() -> None:
    """Configure structlog and the standard logging module based on application settings."""
    # Derive logging configuration from settings, falling back to defaults defined in LoggingConfig.
    config = LoggingConfig(
        level="DEBUG" if settings.debug else "INFO",
        format="%(message)s",
        json=not settings.debug,
    )

    level = getattr(logging, config.level)
    logging.basicConfig(level=level, format=config.format)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()