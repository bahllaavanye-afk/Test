"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, validates

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

    _VALID_SIGNALS = {"buy", "sell", "hold"}
    _VALID_AB_GROUPS = {"champion", "challenger", "shadow"}

    @validates("release_id")
    def _validate_release_id(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("release_id must be a non‑empty string")
        return value

    @validates("model_name")
    def _validate_model_name(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_name must be a non‑empty string")
        return value

    @validates("version")
    def _validate_version(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("version must be a non‑empty string")
        return value

    @validates("symbol")
    def _validate_symbol(self, key: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non‑empty string")
        return value

    @validates("ts")
    def _validate_ts(self, key: str, value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise ValueError("ts must be a datetime instance")
        return value

    @validates("prediction")
    def _validate_prediction(self, key: str, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("prediction must be a numeric value")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("prediction must be between 0 and 1 inclusive")
        return numeric

    @validates("signal")
    def _validate_signal(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("signal must be a string")
        if value not in self._VALID_SIGNALS:
            raise ValueError(f"signal must be one of {self._VALID_SIGNALS}")
        return value

    @validates("confidence")
    def _validate_confidence(self, key: str, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("confidence must be a numeric value")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("confidence must be between 0 and 1 inclusive")
        return numeric

    @validates("latency_ms")
    def _validate_latency_ms(self, key: str, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("latency_ms must be a numeric value")
        if numeric < 0:
            raise ValueError("latency_ms cannot be negative")
        return numeric

    @validates("ab_group")
    def _validate_ab_group(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("ab_group must be a string")
        if value not in self._VALID_AB_GROUPS:
            raise ValueError(f"ab_group must be one of {self._VALID_AB_GROUPS}")
        return value

    @validates("actual_return")
    def _validate_actual_return(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError("actual_return must be a numeric value or None")

    @validates("is_correct")
    def _validate_is_correct(self, key: str, value: bool | None) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        raise ValueError("is_correct must be a boolean value or None")