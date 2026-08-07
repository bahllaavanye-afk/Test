"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


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
    actual_return: Mapped[float | None] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[bool | None] = mapped_column(Boolean)

    def __init__(
        self,
        release_id: str,
        model_name: str,
        version: str,
        symbol: str,
        ts: datetime,
        prediction: float,
        signal: str,
        confidence: float,
        latency_ms: float,
        ab_group: str,
        actual_return: Optional[float] = None,
        is_correct: Optional[bool] = None,
        id: Optional[str] = None,
    ) -> None:
        # Validate string identifiers
        if not isinstance(release_id, str) or not release_id:
            raise ValueError("release_id must be a non-empty string")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty string")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(ab_group, str) or not ab_group:
            raise ValueError("ab_group must be a non-empty string")

        # Validate datetime
        if not isinstance(ts, datetime):
            raise ValueError("ts must be a datetime instance")

        # Validate numeric fields
        if not isinstance(prediction, (float, int)):
            raise ValueError("prediction must be a numeric type")
        if not 0.0 <= float(prediction) <= 1.0:
            raise ValueError("prediction must be within [0, 1]")

        if signal not in {"buy", "sell", "hold"}:
            raise ValueError("signal must be one of 'buy', 'sell', or 'hold'")

        if not isinstance(confidence, (float, int)):
            raise ValueError("confidence must be a numeric type")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

        if not isinstance(latency_ms, (float, int)):
            raise ValueError("latency_ms must be a numeric type")
        if float(latency_ms) < 0:
            raise ValueError("latency_ms cannot be negative")

        if actual_return is not None and not isinstance(actual_return, (float, int)):
            raise ValueError("actual_return must be a numeric type if provided")

        if is_correct is not None and not isinstance(is_correct, bool):
            raise ValueError("is_correct must be a boolean if provided")

        if id is not None and (not isinstance(id, str) or not id):
            raise ValueError("id must be a non-empty string if provided")

        # Assign validated values
        self.id = id  # Let SQLAlchemy generate default if None
        self.release_id = release_id
        self.model_name = model_name
        self.version = version
        self.symbol = symbol
        self.ts = ts
        self.prediction = float(prediction)
        self.signal = signal
        self.confidence = float(confidence)
        self.latency_ms = float(latency_ms)
        self.ab_group = ab_group
        self.actual_return = float(actual_return) if actual_return is not None else None
        self.is_correct = is_correct