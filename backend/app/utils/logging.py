import structlog
import logging
from app.config import settings


def configure_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

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


def log_metrics(signal_count: int, execution_time: float, pnl: float) -> None:
    """
    Log key trading metrics in a structured format.

    Parameters
    ----------
    signal_count : int
        Number of signals generated in the current run.
    execution_time : float
        Execution time in seconds for the evaluated segment.
    pnl : float
        Profit and loss value for the evaluated segment.
    """
    logger.info(
        "trading_metrics",
        signal_count=signal_count,
        execution_time=execution_time,
        pnl=pnl,
    )