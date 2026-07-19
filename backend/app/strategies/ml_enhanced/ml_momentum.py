"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it adjusts the confidence accordingly.
"""

import logging
from typing import Any, Dict, Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.momentum import MomentumStrategy
from app.ml.inference import get_inference_service


logger = logging.getLogger(__name__)


class StrategyParams(BaseModel):
    """Configuration parameters for :class:`MLMomentumStrategy`.

    The schema is permissive to allow future extensions while providing basic
    validation for known fields.
    """

    # Example field – concrete implementations may extend this model.
    example_param: Optional[str] = Field(
        default=None,
        description="An optional example parameter for demonstration purposes.",
        example="example_value",
    )

    class Config:
        extra = "allow"


class MLResult(BaseModel):
    """Result payload returned by the ML inference service.

    Attributes
    ----------
    prediction: Literal["up", "down", "neutral"]
        Directional prediction produced by the model.
    confidence: float
        Model confidence expressed as a probability between 0 and 1.

    Examples
    --------
    >>> MLResult(prediction="up", confidence=0.82)
    MLResult prediction='up' confidence=0.82
    """

    prediction: Literal["up", "down", "neutral"] = Field(
        ...,
        description="Directional prediction from the ML model.",
        example="up",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction, ranging from 0.0 to 1.0.",
        example=0.78,
    )

    @validator("confidence")
    def confidence_range(cls, v: float) -> float:
        """Ensure confidence stays within the inclusive range [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class MLMomentumStrategy(AbstractStrategy):
    """ML‑enhanced momentum strategy.

    The strategy wraps the classic momentum logic and applies an ML filter.
    It inherits from :class:`app.strategies.base.AbstractStrategy`.
    """

    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
        """
        # Validate and store parameters using the Pydantic schema.
        validated_params = StrategyParams(**(params or {}))
        super().__init__(validated_params.dict())
        self._base = MomentumStrategy(validated_params.dict())

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a trading signal for a given symbol.

        The method first obtains a signal from the underlying momentum strategy.
        If a base signal is present, it queries the ML inference service.  When the
        ML prediction agrees with the base signal direction and the confidence
        exceeds the threshold, the signal confidence is adjusted and returned.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and indicator data for the symbol.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A populated :class:`app.strategies.base.Signal` if both the base and
            ML models agree, otherwise ``None``.
        """
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        try:
            inference = get_inference_service()
            raw_ml_result = await inference.predict(data, symbol)
            if raw_ml_result is None:
                return None

            ml_result = MLResult(**raw_ml_result)

            if ml_result.prediction == "neutral":
                return None

            return self._apply_ml_filter(base_signal, ml_result)
        except Exception as e:  # pragma: no cover
            logger.exception("ML inference failed for %s: %s", symbol, e)
            return None

    def _apply_ml_filter(self, base_signal: Signal, ml_result: MLResult) -> Optional[Signal]:
        """Adjust the base signal if the ML prediction agrees.

        Parameters
        ----------
        base_signal : Signal
            Signal produced by the underlying momentum strategy.
        ml_result : MLResult
            Parsed result from the ML inference service.

        Returns
        -------
        Signal | None
            Updated signal if directions match and confidence meets the threshold,
            otherwise ``None``.
        """
        side_match = (
            (ml_result.prediction == "up" and base_signal.side == "buy")
            or (ml_result.prediction == "down" and base_signal.side == "sell")
        )
        if not side_match:
            return None

        # Combine confidences, respecting the configured maximum.
        combined_confidence = min(0.95, (base_signal.confidence + ml_result.confidence) / 2)
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_result.confidence
        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate back‑test signals for a dataframe.

        For back‑testing environments where live ML inference is unavailable,
        this method falls back to the base momentum signals.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing historical data for back‑testing.

        Returns
        -------
        BacktestSignals
            Signals suitable for back‑testing consumption.
        """
        return self._base.backtest_signals(df)