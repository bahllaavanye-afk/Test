"""ML-filtered breakout strategy."""
import pandas as pd
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutParams(BaseModel):
    """Parameters controlling the ML‑enhanced breakout strategy.

    Attributes
    ----------
    confidence_threshold: float
        Minimum confidence required from the ML model for its prediction to be
        considered when adjusting the base breakout signal. Must be between 0 and
        1. Example: ``0.70``.
    max_combined_confidence: float
        Upper bound applied to the combined confidence after merging the base
        signal confidence with the ML confidence. Must be between 0 and 1 and
        greater than or equal to ``confidence_threshold``. Example: ``0.90``.
    """

    confidence_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for ML prediction to adjust the base signal.",
        example=0.70,
    )
    max_combined_confidence: float = Field(
        0.92,
        ge=0.0,
        le=1.0,
        description="Maximum confidence after merging base and ML confidences.",
        example=0.90,
    )

    @validator("max_combined_confidence")
    def max_not_lower_than_threshold(cls, v: float, values: Dict[str, Any]) -> float:
        """Ensure ``max_combined_confidence`` is not lower than ``confidence_threshold``."""
        threshold = values.get("confidence_threshold")
        if threshold is not None and v < threshold:
            raise ValueError(
                "max_combined_confidence must be greater than or equal to confidence_threshold"
            )
        return v


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # Normalise incoming params to a validated Pydantic model.
        if isinstance(params, dict):
            validated_params = MLBreakoutParams(**params)
        elif isinstance(params, MLBreakoutParams):
            validated_params = params
        else:
            validated_params = MLBreakoutParams()

        # Store the validated parameters for later use.
        self.params = validated_params

        # Initialise the parent class with a plain dict representation.
        super().__init__(self.params.dict())

        # Initialise the underlying breakout strategy using the same parameter set.
        self._base = BreakoutStrategy(self.params.dict())

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
                # Combine confidences while respecting the configured upper bound.
                combined_confidence = (base_signal.confidence + ml_result["confidence"]) / 2
                base_signal.confidence = min(self.params.max_combined_confidence, combined_confidence)

                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # On any error, fall back to the base breakout signal unchanged.
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)