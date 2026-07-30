from datetime import datetime, timezone
import logging
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base  # noqa: F401 — re-export for all models

logger = logging.getLogger(__name__)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    @staticmethod
    def log_metrics(signal_count: int, execution_time: float, pnl: float) -> None:
        """
        Log key performance metrics for monitoring purposes.

        Args:
            signal_count: Number of signals processed.
            execution_time: Execution time in seconds.
            pnl: Profit and loss value.
        """
        logger.info(
            "Trading metrics",
            extra={
                "signal_count": signal_count,
                "execution_time": execution_time,
                "pnl": pnl,
            },
        )