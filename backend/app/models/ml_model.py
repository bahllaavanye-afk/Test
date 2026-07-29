import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.database import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # lstm|xgboost|lorentzian|tft|ensemble
    market_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # equity|crypto|polymarket
    symbol: Mapped[Optional[str]] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparams: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
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

    @validates("name", "model_type", "market_type", "artifact_path")
    def _validate_non_empty_string(self, key: str, value: Optional[str]) -> str:
        """Ensure required string fields are non‑empty and not None."""
        if value is None or not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non‑empty string")
        return value.strip()

    @validates("version")
    def _validate_version(self, key: str, value: Optional[int]) -> int:
        """Version must be a positive integer; default to 1 if None or invalid."""
        if not isinstance(value, int) or value < 1:
            return 1
        return value

    @validates("hyperparams")
    def _validate_hyperparams(self, key: str, value: Optional[Dict]) -> Dict:
        """Replace None or non‑dict hyperparams with an empty dict."""
        if not isinstance(value, dict):
            return {}
        return value

    @validates("features")
    def _validate_features(self, key: str, value: Optional[List]) -> List:
        """Replace None or non‑list features with an empty list."""
        if not isinstance(value, list):
            return []
        return value

    @classmethod
    def create(
        cls,
        name: str,
        model_type: str,
        market_type: str,
        artifact_path: str,
        *,
        symbol: Optional[str] = None,
        version: Optional[int] = None,
        hyperparams: Optional[Dict] = None,
        features: Optional[List] = None,
        train_start: Optional[datetime] = None,
        train_end: Optional[datetime] = None,
        val_accuracy: Optional[float] = None,
        val_sharpe: Optional[float] = None,
        val_loss: Optional[float] = None,
        is_active: bool = False,
        trained_at: Optional[datetime] = None,
    ) -> "MLModel":
        """
        Factory method that safely constructs an MLModel instance,
        handling None/empty inputs and applying sensible defaults.
        """
        if trained_at is None:
            trained_at = datetime.utcnow()

        instance = cls(
            name=name,
            model_type=model_type,
            market_type=market_type,
            artifact_path=artifact_path,
            symbol=symbol,
            version=version,
            hyperparams=hyperparams,
            features=features,
            train_start=train_start,
            train_end=train_end,
            val_accuracy=val_accuracy,
            val_sharpe=val_sharpe,
            val_loss=val_loss,
            is_active=is_active,
            trained_at=trained_at,
        )
        return instance

    def increment_version(self) -> None:
        """Safely increment the model version, guarding against overflow."""
        if self.version is None or not isinstance(self.version, int):
            self.version = 1
        else:
            self.version = self.version + 1


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("ml_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)  # up|down|neutral
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    feature_values: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")

    @validates("prediction")
    def _validate_prediction(self, key: str, value: Optional[str]) -> str:
        """Ensure prediction is one of the allowed categories."""
        allowed = {"up", "down", "neutral"}
        if not isinstance(value, str) or value.lower() not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
        return value.lower()

    @validates("confidence")
    def _validate_confidence(self, key: str, value: Optional[float]) -> float:
        """Confidence must be between 0 and 1; clamp if out of bounds."""
        if value is None:
            return 0.0
        try:
            val = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, val))

    @validates("feature_values")
    def _validate_feature_values(self, key: str, value: Optional[Dict]) -> Dict:
        """Replace None or non‑dict feature values with an empty dict."""
        if not isinstance(value, dict):
            return {}
        return value

    @classmethod
    def create(
        cls,
        model_id: str,
        symbol: str,
        ts: datetime,
        prediction: str,
        confidence: float,
        *,
        feature_values: Optional[Dict] = None,
        actual_outcome: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> "MLPrediction":
        """
        Factory method that safely constructs an MLPrediction instance,
        handling None/empty inputs and applying sensible defaults.
        """
        if created_at is None:
            created_at = datetime.utcnow()

        instance = cls(
            model_id=model_id,
            symbol=symbol,
            ts=ts,
            prediction=prediction,
            confidence=confidence,
            feature_values=feature_values,
            actual_outcome=actual_outcome,
            created_at=created_at,
        )
        return instance