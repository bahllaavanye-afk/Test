"""Defines the InferenceLog ORM model which records each prediction made by a serving model.

The model is immutable after creation except for the ``actual_return`` and ``is_correct``
fields, which are populated later via the ``POST /releases/{id}/record-outcome`` endpoint
to enable live accuracy calculations.
"""

import uuid
from datetime import datetime
from typing import ClassVar, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    Attributes
    ----------
    id: str
        Primary key; generated as a UUID string.
    release_id: str
        Foreign key referencing the model release that produced the inference.
    model_name: str
        Name of the model used for inference.
    version: str
        Version identifier of the model.
    symbol: str
        Trading symbol for which the inference was made.
    ts: datetime
        Timestamp of the inference (UTC).
    prediction: float
        Raw model output in the range ``[0, 1]``.
    signal: str
        Discretised trading signal (``buy``, ``sell`` or ``hold``).
    confidence: float
        Calibration metric calculated as ``abs(prediction - 0.5) * 2``.
    latency_ms: float
        Inference latency in milliseconds.
    ab_group: str
        Identifier of the A/B test branch that served the request
        (e.g., ``champion``, ``challenger`` or ``shadow``).
    actual_return: Optional[float]
        Actual market return, populated after the fact.
    is_correct: Optional[bool]
        Indicator whether the prediction was correct, populated after the fact.
    """

    __tablename__: ClassVar[str] = "inference_logs"
    __table_args__: ClassVar[tuple] = (
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
    prediction: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    signal: Mapped[str] = mapped_column(String(8), nullable=False)  # buy|sell|hold
    confidence: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(8, 3), nullable=False)
    ab_group: Mapped[str] = mapped_column(String(16), nullable=False)  # champion|challenger|shadow
    actual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)