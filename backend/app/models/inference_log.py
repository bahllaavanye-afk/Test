"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# Table and index names
TABLE_NAME = "inference_logs"
INDEX_RELEASE_TS = "ix_inf_release_ts"
INDEX_MODEL_SYMBOL = "ix_inf_model_symbol"

# Column length limits
MODEL_NAME_MAX_LEN = 64
VERSION_MAX_LEN = 32
SYMBOL_MAX_LEN = 32
SIGNAL_MAX_LEN = 8
AB_GROUP_MAX_LEN = 16

# Numeric precision specifications (precision, scale)
PREDICTION_PRECISION = (10, 6)
CONFIDENCE_PRECISION = (6, 4)
LATENCY_PRECISION = (8, 3)

# Enumerated string options
SIGNAL_CHOICES = ("buy", "sell", "hold")
AB_GROUP_CHOICES = ("champion", "challenger", "shadow")


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    actual_return and is_correct are filled in after-the-fact via
    POST /releases/{id}/record-outcome so accuracy can be computed live.
    """
    __tablename__ = TABLE_NAME
    __table_args__ = (
        Index(INDEX_RELEASE_TS, "release_id", "ts"),
        Index(INDEX_MODEL_SYMBOL, "model_name", "symbol"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    release_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("model_releases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(MODEL_NAME_MAX_LEN), nullable=False)
    version: Mapped[str] = mapped_column(String(VERSION_MAX_LEN), nullable=False)
    symbol: Mapped[str] = mapped_column(String(SYMBOL_MAX_LEN), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    # Raw model output in [0, 1]
    prediction: Mapped[float] = mapped_column(Numeric(*PREDICTION_PRECISION), nullable=False)
    # Discretised trading signal
    signal: Mapped[str] = mapped_column(String(SIGNAL_MAX_LEN), nullable=False)  # buy|sell|hold
    # Calibration metric: abs(pred - 0.5) * 2
    confidence: Mapped[float] = mapped_column(Numeric(*CONFIDENCE_PRECISION), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(*LATENCY_PRECISION), nullable=False)
    # Which branch of the A/B test served this request
    ab_group: Mapped[str] = mapped_column(String(AB_GROUP_MAX_LEN), nullable=False)  # champion|challenger|shadow
    # Filled in ex-post when actual market return is known
    actual_return: Mapped[float | None] = mapped_column(Numeric(*PREDICTION_PRECISION))
    is_correct: Mapped[bool | None] = mapped_column(Boolean)