import uuid
from datetime import datetime
from sqlalchemy import String, Numeric, DateTime, Boolean, Integer, JSON, ForeignKey, create_engine
from sqlalchemy.orm import Mapped, mapped_column, relationship, sessionmaker
from app.database import Base


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

    predictions: Mapped[list["MLPrediction"]] = relationship("MLPrediction", back_populates="model", cascade="all, delete-orphan")


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


# -------------------- Unit Tests --------------------
# These tests focus on boundary conditions and edge‑case behavior of the ORM models.
# They are deliberately lightweight and use an in‑memory SQLite database.

engine = create_engine("sqlite:///:memory:", echo=False, future=True)
Session = sessionmaker(bind=engine, future=True)
Base.metadata.create_all(engine)


def test_default_hyperparams_and_features_are_independent():
    """Ensure mutable defaults are not shared between instances."""
    session = Session()
    model_a = MLModel(
        name="ModelA",
        model_type="lstm",
        market_type="equity",
        artifact_path="/tmp/model_a.pkl",
        trained_at=datetime.utcnow(),
    )
    model_b = MLModel(
        name="ModelB",
        model_type="xgboost",
        market_type="crypto",
        artifact_path="/tmp/model_b.pkl",
        trained_at=datetime.utcnow(),
    )
    session.add_all([model_a, model_b])
    session.commit()

    # Mutate the defaults of model_a
    model_a.hyperparams["learning_rate"] = 0.01
    model_a.features.append("price")
    session.commit()

    # Refresh from DB to ensure isolation
    session.refresh(model_b)
    assert model_b.hyperparams == {}
    assert model_b.features == []

    session.close()


def test_boundary_numeric_fields():
    """Validate that numeric columns accept their extreme precision and range."""
    session = Session()
    model = MLModel(
        name="BoundaryModel",
        model_type="tft",
        market_type="polymarket",
        artifact_path="/tmp/boundary.pkl",
        trained_at=datetime.utcnow(),
        val_accuracy=1.0000,          # max for Numeric(6,4)
        val_sharpe=9999.9999,         # max for Numeric(8,4)
        val_loss=0.000001,            # min for Numeric(12,6)
    )
    session.add(model)
    session.commit()
    session.refresh(model)

    assert float(model.val_accuracy) == 1.0
    assert float(model.val_sharpe) == 9999.9999
    assert float(model.val_loss) == 0.000001

    session.close()


def test_cascade_delete_predictions():
    """When an MLModel is deleted, associated MLPrediction rows should be removed."""
    session = Session()
    model = MLModel(
        name="CascadeModel",
        model_type="ensemble",
        market_type="equity",
        artifact_path="/tmp/cascade.pkl",
        trained_at=datetime.utcnow(),
    )
    prediction = MLPrediction(
        model=model,
        symbol="AAPL",
        ts=datetime.utcnow(),
        prediction="up",
        confidence=0.85,
        created_at=datetime.utcnow(),
    )
    session.add_all([model, prediction])
    session.commit()

    # Verify both rows exist
    assert session.query(MLModel).filter_by(id=model.id).one_or_none() is not None
    assert session.query(MLPrediction).filter_by(id=prediction.id).one_or_none() is not None

    # Delete the model and ensure the prediction is also removed
    session.delete(model)
    session.commit()

    assert session.query(MLModel).filter_by(id=model.id).one_or_none() is None
    assert session.query(MLPrediction).filter_by(id=prediction.id).one_or_none() is None

    session.close()