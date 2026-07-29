import uuid
import logging
from datetime import datetime, timezone
from sqlalchemy import (
    String,
    Numeric,
    DateTime,
    Boolean,
    Integer,
    JSON,
    ForeignKey,
    event,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
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

    predictions: Mapped[list["MLPrediction"]] = relationship(
        "MLPrediction", back_populates="model", cascade="all, delete-orphan"
    )

    def log_metrics(self, session: Session) -> None:
        """
        Log key performance metrics for the model.

        Metrics:
        - signal_count: total number of predictions generated for this model.
        - avg_execution_latency_ms: average time between prediction creation and now.
        - total_confidence: sum of confidence values, used as a proxy for P&L.
        """
        now = datetime.now(timezone.utc)

        agg = (
            session.query(
                func.count(MLPrediction.id).label("signal_count"),
                func.avg(
                    func.extract(
                        "epoch",
                        func.age(now, MLPrediction.created_at),
                    )
                ).label("avg_latency_seconds"),
                func.sum(MLPrediction.confidence).label("total_confidence"),
            )
            .filter(MLPrediction.model_id == self.id)
            .one()
        )

        signal_count = agg.signal_count or 0
        avg_latency_ms = (agg.avg_latency_seconds or 0) * 1000
        total_confidence = agg.total_confidence or 0.0

        logger.info(
            "MLModel metrics",
            extra={
                "model_id": self.id,
                "model_name": self.name,
                "signal_count": signal_count,
                "avg_execution_latency_ms": round(avg_latency_ms, 2),
                "total_confidence": round(float(total_confidence), 4),
            },
        )


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
    actual_outcome: Mapped[str | None] = mapped_column(String(8))        # filled in ex-post
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    model: Mapped["MLModel"] = relationship("MLModel", back_populates="predictions")


def _after_insert_prediction(mapper, connection, target: MLPrediction):
    """
    SQLAlchemy event hook triggered after a new prediction row is inserted.
    Logs a concise entry and triggers model-level metric aggregation.
    """
    logger.info(
        "New MLPrediction inserted",
        extra={
            "prediction_id": target.id,
            "model_id": target.model_id,
            "symbol": target.symbol,
            "timestamp": target.ts.isoformat(),
            "confidence": float(target.confidence),
        },
    )
    # Use a new Session bound to the same connection to compute aggregated metrics
    sess = Session(bind=connection)
    try:
        model = sess.get(MLModel, target.model_id)
        if model:
            model.log_metrics(sess)
    finally:
        sess.close()


event.listen(MLPrediction, "after_insert", _after_insert_prediction)