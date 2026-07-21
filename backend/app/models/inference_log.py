"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
import logging
from collections import defaultdict
from datetime import datetime

from sqlalchemy import String, Numeric, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

logger = logging.getLogger(__name__)


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    actual_return and is_correct are filled in after-the-fact via
    POST /releases/{id}/record-outcome so accuracy can be computed live.
    """
    __tablename__ = "inference_logs"
    __table_args__ = (
        Index("ix_inf_release_ts", "release_id", "ts"),
        Index("ix_inf_model_symbol", "model_name", "symbol"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    release_id: Mapped[str] = mapped_column(
        String, ForeignKey("model_releases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    # Raw model output in [0, 1]
    prediction: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    # Discretised trading signal
    signal: Mapped[str] = mapped_column(String(8), nullable=False)       # buy|sell|hold
    # Calibration metric: abs(pred - 0.5) * 2
    confidence: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(8, 3), nullable=False)
    # Which branch of the A/B test served this request
    ab_group: Mapped[str] = mapped_column(String(16), nullable=False)    # champion|challenger|shadow
    # Filled in ex-post when actual market return is known
    actual_return: Mapped[float | None] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[bool | None] = mapped_column(Boolean)

    # Class‑level aggregates for structured logging
    _signal_counts: defaultdict = defaultdict(int)
    _total_latency: float = 0.0
    _cumulative_pnl: float = 0.0
    _total_records: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Update aggregates
        InferenceLog._signal_counts[self.signal] += 1
        InferenceLog._total_latency += float(self.latency_ms)
        if self.actual_return is not None:
            InferenceLog._cumulative_pnl += float(self.actual_return)
        InferenceLog._total_records += 1

        # Structured log at INFO level
        logger.info(
            "InferenceLog recorded",
            extra={
                "signal": self.signal,
                "latency_ms": float(self.latency_ms),
                "actual_return": float(self.actual_return) if self.actual_return is not None else None,
                "signal_counts": dict(InferenceLog._signal_counts),
                "avg_latency_ms": InferenceLog._total_latency / InferenceLog._total_records,
                "cumulative_pnl": InferenceLog._cumulative_pnl,
                "total_records": InferenceLog._total_records,
            },
        )