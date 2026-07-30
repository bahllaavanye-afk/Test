import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from sqlalchemy import String, Numeric, DateTime, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)   # lstm|xgboost|lorentzian|tft|ensemble
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)  # equity|crypto|polymarket
    symbol: Mapped[Optional[str]] = mapped_column(String(32))               # None = multi-symbol
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparams: Mapped[Dict] = mapped_column(JSON, default=dict)
    features: Mapped[List] = mapped_column(JSON, default=list)
    train_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    val_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    val_loss: Mapped[Optional[float]] = mapped_column(Numeric(12, 6))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    predictions: Mapped[List["MLPrediction"]] = relationship("MLPrediction", back_populates="model")

    # ---------------------------------------------------------------------
    # Strategy helpers
    # ---------------------------------------------------------------------
    def recent_predictions(
        self,
        lookback: timedelta,
        direction: Optional[str] = None,
    ) -> List["MLPrediction"]:
        """
        Return predictions for this model within the given lookback window.
        Optionally filter by prediction direction (up/down/neutral).
        """
        cutoff = datetime.utcnow() - lookback
        return [
            p
            for p in self.predictions
            if p.ts >= cutoff and (direction is None or p.prediction == direction)
        ]

    def generate_entry_signals(
        self,
        confidence_threshold: float = 0.70,
        confirmation_window: timedelta = timedelta(minutes=5),
        min_confirmations: int = 2,
    ) -> List[Tuple[datetime, str, float]]:
        """
        Produce entry signals that satisfy tighter entry criteria.

        Criteria:
        1. Prediction confidence must exceed ``confidence_threshold``.
        2. The same prediction direction must appear at least ``min_confirmations``
           times within ``confirmation_window``.
        3. The model must be active.

        Returns a list of tuples (timestamp, direction, confidence) representing
        the earliest timestamp that satisfies the criteria.
        """
        if not self.is_active:
            return []

        # Sort predictions chronologically for reliable windowing
        sorted_preds = sorted(self.predictions, key=lambda p: p.ts)

        signals: List[Tuple[datetime, str, float]] = []
        i = 0
        while i < len(sorted_preds):
            base_pred = sorted_preds[i]
            if base_pred.confidence < confidence_threshold:
                i += 1
                continue

            # Gather confirmations within the window
            window_end = base_pred.ts + confirmation_window
            confirmations = [
                p
                for p in sorted_preds[i + 1 :]
                if p.ts <= window_end and p.prediction == base_pred.prediction
            ]

            # Include the base prediction as the first confirmation
            total_confirmations = 1 + len(confirmations)

            if total_confirmations >= min_confirmations:
                # Use the timestamp of the earliest prediction that meets the criteria
                earliest_ts = base_pred.ts
                signals.append((earliest_ts, base_pred.prediction, base_pred.confidence))
                # Skip ahead past the window to avoid overlapping signals
                i = next(
                    (idx for idx, p in enumerate(sorted_preds) if p.ts > window_end), len(sorted_preds)
                )
            else:
                i += 1

        return signals

    def generate_exit_signals(
        self,
        active_entries: List[Tuple[datetime, str]],
        exit_confidence_drop: float = 0.15,
        max_holding_period: timedelta = timedelta(hours=4),
    ) -> List[datetime]:
        """
        Determine exit timestamps for open positions.

        Exit criteria:
        1. Confidence falls by ``exit_confidence_drop`` relative to entry confidence.
        2. Position exceeds ``max_holding_period``.
        3. Prediction direction flips.

        Parameters
        ----------
        active_entries: List of tuples (entry_timestamp, direction) for open trades.
        exit_confidence_drop: Minimum drop in confidence to trigger an exit.
        max_holding_period: Upper bound on trade duration.

        Returns
        -------
        List of timestamps at which exits should be executed.
        """
        exits: List[datetime] = []
        # Index predictions by timestamp for quick lookup
        preds_by_ts = {p.ts: p for p in self.predictions}

        for entry_ts, direction in active_entries:
            # Find the prediction at entry time to capture entry confidence
            entry_pred = preds_by_ts.get(entry_ts)
            if not entry_pred:
                continue

            # Scan forward in time to evaluate exit conditions
            for pred in sorted(self.predictions, key=lambda p: p.ts):
                if pred.ts <= entry_ts:
                    continue

                # Time-based exit
                if pred.ts - entry_ts >= max_holding_period:
                    exits.append(pred.ts)
                    break

                # Direction flip exit
                if pred.prediction != direction and pred.prediction != "neutral":
                    exits.append(pred.ts)
                    break

                # Confidence drop exit
                if entry_pred.confidence - pred.confidence >= exit_confidence_drop:
                    exits.append(pred.ts)
                    break

        return exits


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
    feature_values: Mapped[Dict] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(8))        # filled in ex-post
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")

    # ---------------------------------------------------------------------
    # Convenience utilities
    # ---------------------------------------------------------------------
    def is_strong_signal(self, threshold: float = 0.70) -> bool:
        """
        Determine if this prediction qualifies as a strong signal based on confidence.
        """
        return self.confidence >= threshold and self.prediction in {"up", "down"}

    def as_tuple(self) -> Tuple[datetime, str, float]:
        """
        Return a lightweight representation of the prediction useful for signal pipelines.
        """
        return (self.ts, self.prediction, self.confidence)