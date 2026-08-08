"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field, validator
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InferenceLog(Base):
    """
    Immutable record of a single model inference.

    `actual_return` and `is_correct` are filled in after‑the‑fact via
    `POST /releases/{id}/record-outcome` so accuracy can be computed live.
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
    # Filled in ex‑post when actual market return is known
    actual_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean)


class InferenceLogSchema(BaseModel):
    """
    Pydantic schema for serialising and validating `InferenceLog` records.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the inference record.",
        example="a1b2c3d4-5678-90ab-cdef-1234567890ab",
    )
    release_id: str = Field(
        ...,
        description="Foreign key linking to the model release used for inference.",
        example="release-2024-08-01",
    )
    model_name: str = Field(
        ...,
        description="Name of the model that generated the inference.",
        example="price_predictor",
        max_length=64,
    )
    version: str = Field(
        ...,
        description="Version string of the model.",
        example="v1.2.3",
        max_length=32,
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol the inference was made for.",
        example="AAPL",
        max_length=32,
    )
    ts: datetime = Field(
        ...,
        description="Timestamp (UTC) when the inference was generated.",
        example="2024-08-01T12:34:56.789Z",
    )
    prediction: float = Field(
        ...,
        description="Raw model output, a probability in the inclusive range [0, 1].",
        example=0.732,
        ge=0.0,
        le=1.0,
    )
    signal: Literal["buy", "sell", "hold"] = Field(
        ...,
        description="Discretised trading signal derived from the prediction.",
        example="buy",
    )
    confidence: float = Field(
        ...,
        description="Calibration metric calculated as `abs(prediction - 0.5) * 2`, also in [0, 1].",
        example=0.464,
        ge=0.0,
        le=1.0,
    )
    latency_ms: float = Field(
        ...,
        description="Model serving latency in milliseconds.",
        example=12.345,
        ge=0.0,
    )
    ab_group: Literal["champion", "challenger", "shadow"] = Field(
        ...,
        description="A/B test group that served the request.",
        example="champion",
    )
    actual_return: Optional[float] = Field(
        None,
        description="Realised market return after the inference, recorded post‑trade.",
        example=0.015,
    )
    is_correct: Optional[bool] = Field(
        None,
        description="Whether the prediction was correct in hindsight.",
        example=True,
    )

    @validator("prediction")
    def check_prediction_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("prediction must be between 0 and 1 inclusive")
        return v

    @validator("confidence")
    def check_confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1 inclusive")
        return v

    @validator("latency_ms")
    def check_latency_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("latency_ms must be non‑negative")
        return v

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
                "release_id": "release-2024-08-01",
                "model_name": "price_predictor",
                "version": "v1.2.3",
                "symbol": "AAPL",
                "ts": "2024-08-01T12:34:56.789Z",
                "prediction": 0.732,
                "signal": "buy",
                "confidence": 0.464,
                "latency_ms": 12.345,
                "ab_group": "champion",
                "actual_return": None,
                "is_correct": None,
            }
        }