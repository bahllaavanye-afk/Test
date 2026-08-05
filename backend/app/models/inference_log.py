"""InferenceLog ORM — records every prediction made by a serving model."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# ----------------------------------------------------------------------
# SQLAlchemy ORM model
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Pydantic schema for API interaction
# ----------------------------------------------------------------------
from pydantic import BaseModel, Field, validator


class InferenceLogSchema(BaseModel):
    """Schema representing a single inference record for API payloads."""

    id: str = Field(
        ...,
        description="Unique identifier for the inference record.",
        example="d4f5a3e2-9b1c-4f7a-8c2e-6a1b2c3d4e5f",
    )
    release_id: str = Field(
        ...,
        description="Foreign key referencing the model release that generated the inference.",
        example="release_2023_09_15",
    )
    model_name: str = Field(
        ...,
        description="Name of the model that produced the inference.",
        example="price_predictor",
    )
    version: str = Field(
        ...,
        description="Version string of the model.",
        example="v1.2.0",
    )
    symbol: str = Field(
        ...,
        description="Trading symbol (ticker) the inference pertains to.",
        example="AAPL",
    )
    ts: datetime = Field(
        ...,
        description="Timestamp when the inference was made (UTC).",
        example="2023-09-15T14:30:00Z",
    )
    prediction: float = Field(
        ...,
        description="Raw model output representing probability or score, constrained to [0, 1].",
        example=0.73,
        ge=0.0,
        le=1.0,
    )
    signal: str = Field(
        ...,
        description="Discretised trading signal derived from the prediction.",
        example="buy",
    )
    confidence: float = Field(
        ...,
        description="Calibration metric: `abs(prediction - 0.5) * 2`, ranges from 0 to 1.",
        example=0.46,
        ge=0.0,
        le=1.0,
    )
    latency_ms: float = Field(
        ...,
        description="Model inference latency in milliseconds.",
        example=12.345,
        ge=0.0,
    )
    ab_group: str = Field(
        ...,
        description="A/B test group that served the request.",
        example="champion",
    )
    actual_return: Optional[float] = Field(
        None,
        description="Actual market return observed after the inference, populated later.",
        example=0.0012,
    )
    is_correct: Optional[bool] = Field(
        None,
        description="Whether the prediction was correct once actual return is known.",
        example=True,
    )

    @validator("signal")
    def validate_signal(cls, v: str) -> str:
        allowed = {"buy", "sell", "hold"}
        if v not in allowed:
            raise ValueError(f"signal must be one of {allowed}, got {v!r}")
        return v

    @validator("ab_group")
    def validate_ab_group(cls, v: str) -> str:
        allowed = {"champion", "challenger", "shadow"}
        if v not in allowed:
            raise ValueError(f"ab_group must be one of {allowed}, got {v!r}")
        return v

    @validator("prediction", "confidence")
    def validate_probability_range(cls, v: float, field) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{field.name} must be between 0 and 1, got {v}")
        return v

    @validator("latency_ms")
    def validate_latency(cls, v: float) -> float:
        if v < 0:
            raise ValueError("latency_ms must be non‑negative")
        return v

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "d4f5a3e2-9b1c-4f7a-8c2e-6a1b2c3d4e5f",
                "release_id": "release_2023_09_15",
                "model_name": "price_predictor",
                "version": "v1.2.0",
                "symbol": "AAPL",
                "ts": "2023-09-15T14:30:00Z",
                "prediction": 0.73,
                "signal": "buy",
                "confidence": 0.46,
                "latency_ms": 12.345,
                "ab_group": "champion",
                "actual_return": None,
                "is_correct": None,
            }
        }