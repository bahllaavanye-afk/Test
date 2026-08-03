import uuid
import logging
from datetime import datetime
from sqlalchemy import String, Numeric, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

logger = logging.getLogger(__name__)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    test_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics_history: Mapped[list] = mapped_column(JSON, default=list)  # [{epoch, loss, acc}, ...]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def log_execution_metrics(self, signal_count: int, pnl: float) -> None:
        """
        Logs key execution metrics for the experiment at INFO level.

        Parameters
        ----------
        signal_count: int
            Number of signals generated during the experiment run.
        pnl: float
            Profit and loss achieved for the experiment.
        """
        exec_time_seconds = None
        if self.started_at and self.completed_at:
            try:
                exec_time_seconds = (self.completed_at - self.started_at).total_seconds()
            except Exception as e:
                logger.debug(
                    "Failed to compute execution time for experiment %s: %s",
                    self.id,
                    e,
                )

        logger.info(
            "Experiment execution metrics",
            extra={
                "experiment_id": self.id,
                "experiment_name": self.name,
                "signal_count": signal_count,
                "execution_time_seconds": exec_time_seconds,
                "pnl": pnl,
            },
        )