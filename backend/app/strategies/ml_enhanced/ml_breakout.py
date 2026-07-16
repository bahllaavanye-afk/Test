"""ML-filtered breakout strategy."""
import pandas as pd
from pydantic import BaseModel, Field, validator
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutParams(BaseModel):
    """Parameters controlling the ML‑enhanced breakout strategy.

    Attributes
    ----------
    confidence_threshold: float
        Minimum confidence from the ML model required to augment the base signal.
        Must be between 0 and 1. Example: 0.7.
    max_combined_confidence: float
        Upper bound applied to the blended confidence of base and ML signals.
        Must be between 0 and 1. Example: 0.92.
    """
    confidence_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Minimum confidence from ML model to consider.",
        example=0.7,
    )
    max_combined_confidence: float = Field(
        0.92,
        ge=0.0,
        le=1.0,
        description="Maximum allowed combined confidence after blending.",
        example=0.9,
    )

    @validator("confidence_threshold", "max_combined_confidence")
    def _validate_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("value must be between 0 and 1")
        return v


class MLInferenceResult(BaseModel):
    """Schema for the inference result returned by the ML service.

    Attributes
    ----------
    prediction: str
        Predicted direction; allowed values are ``\"up\"`` or ``\"down\"``.
        Example: ``\"up\"``.
    confidence: float
        Confidence score associated with the prediction. Must be between 0 and 1.
        Example: 0.78.
    """
    prediction: str = Field(
        ...,
        description="Predicted market direction.",
        example="up",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the prediction.",
        example=0.78,
    )

    @validator("prediction")
    def _validate_prediction(cls, v: str) -> str:
        if v not in {"up", "down"}:
            raise ValueError("prediction must be either 'up' or 'down'")
        return v

    @validator("confidence")
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Validate and store strategy parameters using the Pydantic schema.
        self.params = MLBreakoutParams.parse_obj(params or {})
        # Pass original params to the underlying manual breakout strategy.
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            raw_ml_result = await inference.predict(data, symbol)
            # Validate the ML inference payload.
            ml_result = MLInferenceResult(**raw_ml_result)
            if (
                ml_result.confidence > self.params.confidence_threshold
                and ml_result.prediction == "up"
            ):
                blended_conf = (base_signal.confidence + ml_result.confidence) / 2
                base_signal.confidence = min(
                    self.params.max_combined_confidence, blended_conf
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # Preserve the base signal if ML augmentation fails.
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)