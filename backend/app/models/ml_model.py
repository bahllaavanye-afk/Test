"""ORM models for machine‑learning models and their predictions.

This module defines the SQLAlchemy ``Base`` subclasses used throughout the
platform to store trained models and the predictions they generate.
"""

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
    """SQLAlchemy model representing a trained machine‑learning model.

    Attributes
    ----------
    id : Mapped[str]
        Primary‑key UUID string.
    name : Mapped[str]
        Human‑readable name of the model.
    model_type : Mapped[str]
        Type of model (e.g., ``'lstm'``, ``'xgboost'``, ``'lorentzian'``, ``'tft'``,
        ``'ensemble'``).
    market_type : Mapped[str]
        Market the model is built for (e.g., ``'equity'``, ``'crypto'``,
        ``'polymarket'``).
    symbol : Mapped[Optional[str]]
        Specific symbol the model targets, or ``None`` for a multi‑symbol model.
    version : Mapped[int]
        Incremental version number of the model.
    artifact_path : Mapped[str]
        Filesystem path to the serialized model artifact.
    hyperparams : Mapped[dict]
        Hyper‑parameter dictionary stored as JSON.
    features : Mapped[list]
        List of feature names used by the model.
    train_start : Mapped[Optional[datetime]]
        Timestamp marking the start of the training period.
    train_end : Mapped[Optional[datetime]]
        Timestamp marking the end of the training period.
    val_accuracy : Mapped[Optional[float]]
        Validation accuracy metric.
    val_sharpe : Mapped[Optional[float]]
        Validation Sharpe ratio.
    val_loss : Mapped[Optional[float]]
        Validation loss.
    is_active : Mapped[bool]
        Flag indicating whether the model is currently active.
    trained_at : Mapped[datetime]
        Timestamp when the model was trained.
    predictions : Mapped[List[\"MLPrediction\"]]
        Relationship to :class:`MLPrediction` objects.
    """

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
        """Return the most recent prediction based on timestamp.

        Returns
        -------
        Optional[MLPrediction]
            The prediction with the highest ``ts`` value, or ``None`` if no
            predictions are associated with the model.
        """
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.ts)

    def is_signal_strong(self, recent: List["MLPrediction"]) -> bool:
        """Determine whether a recent set of predictions satisfies entry criteria.

        The signal is considered strong when:

        * The latest prediction confidence exceeds :data:`ENTRY_CONFIDENCE_THRESHOLD`.
        * The latest prediction direction is not ``'neutral'``.
        * At least :data:`CONFIRMATION_COUNT` of the most recent predictions share
          the same direction as the latest one.

        Parameters
        ----------
        recent : List[MLPrediction]
            A list of recent predictions to evaluate.

        Returns
        -------
        bool
            ``True`` if the entry signal meets all criteria; otherwise ``False``.
        """
        if not recent:
            return False

        latest = max(recent, key=lambda p: p.ts)
        if latest.prediction == "neutral" or latest.confidence < ENTRY_CONFIDENCE_THRESHOLD:
            return False

        same_dir_count = 0
        for pred in sorted(recent, key=lambda p: p.ts, reverse=True):
            if pred.prediction == latest.prediction:
                same_dir_count += 1
                if same_dir_count >= CONFIRMATION_COUNT:
                    return True
            else:
                break

        return False

    def should_exit(self, recent: List["MLPrediction"]) -> bool:
        """Evaluate whether the position should be exited based on recent predictions.

        Exit conditions are triggered when:

        * The latest prediction confidence falls below
          ``ENTRY_CONFIDENCE_THRESHOLD - EXIT_CONFIDENCE_DROP``.
        * An opposite prediction direction appears within the supplied recent
          window.

        Parameters
        ----------
        recent : List[MLPrediction]
            A list of recent predictions to evaluate.

        Returns
        -------
        bool
            ``True`` if any exit condition is met; otherwise ``False``.
        """
        if not recent:
            return False

        latest = max(recent, key=lambda p: p.ts)
        confidence_floor = ENTRY_CONFIDENCE_THRESHOLD - EXIT_CONFIDENCE_DROP
        if latest.confidence < confidence_floor:
            return True

        opposite = {"up": "down", "down": "up"}
        opposite_dir = opposite.get(latest.prediction)
        if opposite_dir:
            for pred in recent:
                if pred.prediction == opposite_dir:
                    return True

        return False


class MLPrediction(Base):
    """SQLAlchemy model representing a single prediction generated by an :class:`MLModel`.

    Attributes
    ----------
    id : Mapped[str]
        Primary‑key UUID string.
    model_id : Mapped[str]
        Foreign key linking to the originating :class:`MLModel`.
    symbol : Mapped[str]
        Symbol the prediction applies to.
    ts : Mapped[datetime]
        Timestamp of the prediction (timezone‑aware).
    prediction : Mapped[str]
        Directional prediction – ``'up'``, ``'down'`` or ``'neutral'``.
    confidence : Mapped[float]
        Model confidence for the prediction (0‑1 range).
    feature_values : Mapped[dict]
        Feature values used for this prediction, stored as JSON.
    actual_outcome : Mapped[Optional[str]]
        Realised outcome filled in ex‑post; ``None`` if not yet known.
    created_at : Mapped[datetime]
        Timestamp when the record was created.
    model : Mapped[MLModel]
        Relationship back to the originating model.
    """

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