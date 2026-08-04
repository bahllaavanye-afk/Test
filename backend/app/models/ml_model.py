import uuid
import logging
from datetime import datetime
from sqlalchemy import String, Numeric, DateTime, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

logger = logging.getLogger(__name__)


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)   # lstm|xgboost|lorentzian|tft|ensemble
    market_type: Mapped[str] = mapped_column(String(16), nullable=False)  # equity|crypto|polymarket
    symbol: Mapped[str | None] = mapped_column(String(32))               # None = multi-symbol
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    hyperparams: Mapped[dict] = mapped_column(JSON, default=dict)
    features: Mapped[list] = mapped_column(JSON, default=list)
    train_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    train_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    val_loss: Mapped[float | None] = mapped_column(Numeric(12, 6))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    predictions: Mapped[list["MLPrediction"]] = relationship("MLPrediction", back_populates="model")

    def log_prediction_metrics(self, signal_count: int, execution_time: float, pnl: float) -> None:
        """
        Emit structured log entry with key performance metrics for the model.

        Args:
            signal_count: Number of signals generated for the current run.
            execution_time: Time taken (seconds) to generate predictions.
            pnl: Profit and loss realized from the predictions.
        """
        logger.info(
            "model_prediction_metrics",
            extra={
                "model_id": self.id,
                "model_name": self.name,
                "signal_count": signal_count,
                "execution_time_sec": execution_time,
                "pnl": pnl,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )


class MLPrediction(Base):
    __tablename__ = "ml_predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(String, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)   # up|down|neutral
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    feature_values: Mapped[dict] = mapped_column(JSON, default=dict)
    actual_outcome: Mapped[str | None] = mapped_column(String(8))        # filled in ex-post
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")