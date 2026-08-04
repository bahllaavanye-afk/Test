"""ModelRelease ORM — tracks every trained model artifact through its serving lifecycle."""
import uuid
import logging
from datetime import datetime
from sqlalchemy import String, Float, Integer, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)

class ModelRelease(Base, TimestampMixin):
    """
    One row per trained model artifact registered for serving.

    Lifecycle:
        registered → shadow → challenger → champion → archived

    Only one champion and one challenger per (model_name) are allowed at a time.
    Promoting a challenger to champion atomically archives the old champion.
    """
    __tablename__ = "model_releases"
    __table_args__ = (
        Index("ix_mr_model_status", "model_name", "status"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Logical model name shared across versions, e.g. "lstm_momentum"
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Artifact version, e.g. "v1.0.0" or "20240115_001"
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Filesystem path to the serialized model weights / pickle
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # Serialization framework
    framework: Mapped[str] = mapped_column(String(32), nullable=False, default="pytorch")
    # Model architecture size params
    n_features: Mapped[int | None] = mapped_column(Integer)
    seq_len: Mapped[int | None] = mapped_column(Integer)
    # JSON blobs
    model_params: Mapped[dict] = mapped_column(JSON, default=dict)
    training_config: Mapped[dict] = mapped_column(JSON, default=dict)
    train_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    live_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    # Serving state
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="registered", index=True
    )
    # % of inference traffic routed to this release when it's a challenger (0–100)
    traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Free-text notes from the creator
    notes: Mapped[str | None] = mapped_column(Text)
    # Timestamps for key lifecycle events
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Who registered this release (email or "system")
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    def log_metrics(self, signal_count: int, execution_time: float, pnl: float) -> None:
        """
        Log key runtime metrics for this model release.

        Parameters
        ----------
        signal_count: int
            Number of trading signals generated during the interval.
        execution_time: float
            Total execution time in seconds for the interval.
        pnl: float
            Profit & loss realized for the interval.

        The log is emitted at INFO level with a structured payload to facilitate
        downstream aggregation and monitoring.
        """
        logger.info(
            "model_release_metrics",
            extra={
                "model_release_id": self.id,
                "model_name": self.model_name,
                "version": self.version,
                "signal_count": signal_count,
                "execution_time_seconds": execution_time,
                "pnl": pnl,
                "status": self.status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )