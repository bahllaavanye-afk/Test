"""ML-filtered breakout strategy."""
import pandas as pd
from pydantic import BaseModel, Field, validator
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutParams(BaseModel):
    """Parameters governing the ML‑enhanced breakout strategy.

    Attributes
    ----------
    lookback_window: int
        Number of periods to consider for the underlying breakout detection.
        Must be a positive integer and a multiple of 5.
    confidence_threshold: float
        Minimum ML confidence required before the base signal is adjusted.
        Value must lie in the inclusive range [0, 1].
    max_combined_confidence: float
        Upper bound applied to the combined confidence of the base and ML signals.
        Value must lie in the inclusive range [0, 1].
    """
    lookback_window: int = Field(
        20,
        description="Number of periods to look back for breakout detection.",
        example=20,
        ge=1,
    )
    confidence_threshold: float = Field(
        0.65,
        description="Minimum ML confidence to adjust the base signal.",
        example=0.65,
        ge=0.0,
        le=1.0,
    )
    max_combined_confidence: float = Field(
        0.92,
        description="Upper bound for the combined signal confidence.",
        example=0.92,
        ge=0.0,
        le=1.0,
    )

    @validator("lookback_window")
    def check_lookback_multiple_of_five(cls, v: int) -> int:
        if v % 5 != 0:
            raise ValueError("lookback_window must be a multiple of 5")
        return v

    @validator("confidence_threshold", "max_combined_confidence")
    def check_probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0 and 1 inclusive")
        return v


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        # Validate and store strategy parameters
        validated_params = MLBreakoutParams(**(params or {}))
        super().__init__(validated_params.dict())
        self.params = validated_params
        self._base = BreakoutStrategy(validated_params.dict())

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result["confidence"] > self.params.confidence_threshold
                and ml_result["prediction"] == "up"
            ):
                combined_conf = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(self.params.max_combined_confidence, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)