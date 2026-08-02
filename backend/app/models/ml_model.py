import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from app.database import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # lstm|xgboost|lorentzian|tft|ensemble
    market_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # equity|crypto|polymarket
    symbol: Mapped[Optional[str]] = mapped_column(String(32))  # None = multi-symbol
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparams: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # mutable default handled via callable
    features: Mapped[List[Any]] = mapped_column(JSON, default=list)
    train_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    val_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    val_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 6))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    predictions: Mapped[List["MLPrediction"]] = relationship(
        "MLPrediction", back_populates="model"
    )

    @validates("version")
    def _validate_version(self, key: str, value: int) -> int:
        """Ensure version is a positive integer (off‑by‑one safety)."""
        if value is None:
            return 1
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer, got {value}")
        return value

    @validates("hyperparams")
    def _validate_hyperparams(self, key: str, value: Optional[Dict]) -> Dict:
        """Convert None to empty dict and ensure a dict is stored."""
        return {} if value is None else dict(value)

    @validates("features")
    def _validate_features(self, key: str, value: Optional[List]) -> List:
        """Convert None to empty list and ensure a list is stored."""
        return [] if value is None else list(value)

    def model_age_days(self) -> Optional[int]:
        """Return the age of the model in days, handling missing timestamps."""
        if self.trained_at is None:
            return None
        delta = datetime.utcnow() - self.trained_at.replace(tzinfo=None)
        return max(delta.days, 0)  # guard against negative values


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    model_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    prediction: Mapped[str] = mapped_column(
        String(8), nullable=False
    )  # up|down|neutral
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    feature_values: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(8))  # filled in ex‑post
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")

    @validates("prediction")
    def _validate_prediction(self, key: str, value: str) -> str:
        """Accept only predefined prediction strings; treat empty or None as 'neutral'."""
        allowed = {"up", "down", "neutral"}
        if not value:
            return "neutral"
        val = value.lower()
        if val not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {value}")
        return val

    @validates("confidence")
    def _validate_confidence(self, key: str, value: Optional[float]) -> float:
        """Clamp confidence to [0.0, 1.0]; replace None with 0.0."""
        if value is None:
            return 0.0
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a numeric value, got {value}")
        return max(0.0, min(1.0, val))

    @validates("feature_values")
    def _validate_feature_values(self, key: str, value: Optional[Dict]) -> Dict:
        """Convert None to empty dict and ensure a dict is stored."""
        return {} if value is None else dict(value)