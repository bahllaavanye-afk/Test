"""InferenceLog ORM — records every prediction made by a serving model.

This module defines the :class:`InferenceLog` SQLAlchemy model which stores a
single inference event.  The record is immutable except for the
``actual_return`` and ``is_correct`` columns that are populated after the
market outcome is known.  The model is used by the API layer to log
predictions and later compute live accuracy metrics.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    Attributes
    ----------
    id: str
        Primary key generated as a UUID4 string.
    release_id: str
        Foreign key referencing the model release that generated the inference.
    model_name: str
        Human‑readable name of the model.
    version: str
        Version identifier of the model.
    symbol: str
        Trading symbol the inference was made for.
    ts: datetime
        Timestamp of the inference (UTC, timezone‑aware).
    prediction: float
        Raw model output in the range ``[0, 1]``.
    signal: str
        Discretised trading signal (e.g. ``'buy'``, ``'sell'``, ``'hold'``).
    confidence: float
        Calibration metric calculated as ``abs(prediction - 0.5) * 2``.
    latency_ms: float
        Inference latency in milliseconds.
    ab_group: str
        Identifier of the A/B test branch that served the request
        (e.g. ``'champion'``, ``'challenger'``, ``'shadow'``).
    actual_return: Optional[float]
        Realised market return, populated after the outcome is known.
    is_correct: Optional[bool]
        Flag indicating whether the signal matched the market direction,
        populated after the outcome is known.
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
        String,
        ForeignKey("model_releases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    # Raw model output in [0, 1]
    prediction: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    # Discretised trading signal
    signal: Mapped[str] = mapped_column(String(8), nullable=False)  # buy|sell|hold
    # Calibration metric: abs(pred - 0.5) * 2
    confidence: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(8, 3), nullable=False)
    # Which branch of the A/B test served this request
    ab_group: Mapped[str] = mapped_column(String(16), nullable=False)  # champion|challenger|shadow
    # Filled in ex-post when actual market return is known
    actual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)