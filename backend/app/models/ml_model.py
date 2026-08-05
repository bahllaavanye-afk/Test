import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Numeric, DateTime, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

# Strategy configuration constants
ENTRY_CONFIDENCE_THRESHOLD: float = 0.70  # Minimum confidence for an entry signal
CONFIRMATION_COUNT: int = 3               # Required consecutive predictions in the same direction
EXIT_CONFIDENCE_DROP: float = 0.20        # Confidence drop that triggers an exit


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)   # lstm|xgboost|lorentzian|tft|ensemble
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)  # equity|crypto|polymarket
    symbol: Mapped[Optional[str]] = mapped_column(String(32))               # None = multi-symbol
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparams: Mapped[dict] = mapped_column(JSON, default=dict)
    features: Mapped[list] = mapped_column(JSON, default=list)
    train_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    val_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    val_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 6))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    predictions: Mapped[List["MLPrediction"]] = relationship(
        "MLPrediction", back_populates="model", cascade="all, delete-orphan"
    )

    def latest_prediction(self) -> Optional["MLPrediction"]:
        """Return the most recent prediction based on timestamp."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.ts)

    def is_signal_strong(self, recent: List["MLPrediction"]) -> bool:
        """
        Determine if the entry signal meets tightened criteria.

        Conditions:
        1. Latest prediction confidence exceeds ENTRY_CONFIDENCE_THRESHOLD.
        2. Prediction direction is not 'neutral'.
        3. At least CONFIRMATION_COUNT of the most recent predictions share the same direction.
        """
        if not recent:
            return False

        latest = max(recent, key=lambda p: p.ts)
        if latest.prediction == "neutral" or latest.confidence < ENTRY_CONFIDENCE_THRESHOLD:
            return False

        # Count consecutive predictions with the same direction as the latest
        same_dir_count = 0
        for pred in sorted(recent, key=lambda p: p.ts, reverse=True):
            if pred.prediction == latest.prediction:
                same_dir_count += 1
                if same_dir_count >= CONFIRMATION_COUNT:
                    return True
            else:
                break  # Stop counting once a different direction appears

        return False

    def should_exit(self, recent: List["MLPrediction"]) -> bool:
        """
        Evaluate exit conditions.

        Exit is triggered when:
        * Confidence drops below (ENTRY_CONFIDENCE_THRESHOLD - EXIT_CONFIDENCE_DROP), or
        * An opposite prediction appears within the recent window.
        """
        if not recent:
            return False

        latest = max(recent, key=lambda p: p.ts)
        confidence_floor = ENTRY_CONFIDENCE_THRESHOLD - EXIT_CONFIDENCE_DROP
        if latest.confidence < confidence_floor:
            return True

        # Detect opposite direction within the recent predictions
        opposite = {"up": "down", "down": "up"}
        opposite_dir = opposite.get(latest.prediction)
        if opposite_dir:
            for pred in recent:
                if pred.prediction == opposite_dir:
                    return True

        return False


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)   # up|down|neutral
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    feature_values: Mapped[dict] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(8))        # filled in ex-post
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")