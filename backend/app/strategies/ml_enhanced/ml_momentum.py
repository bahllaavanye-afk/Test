"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it adjusts the confidence accordingly.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.momentum import MomentumStrategy
from app.ml.inference import get_inference_service


logger = logging.getLogger(__name__)


class StrategyParams(BaseModel):
    """Configuration parameters for :class:`MLMomentumStrategy`.

    Attributes
    ----------
    confidence_threshold: float
        Minimum combined confidence required to emit a signal.
        Must be between 0 and 1 (inclusive). Example: ``0.65``.
    tick_interval_seconds: float
        Interval in seconds between strategy ticks. Must be positive.
        Example: ``3600.0``.
    """

    confidence_threshold: float = Field(
        0.65,
        description="Minimum combined confidence required to emit a signal.",
        example=0.65,
        ge=0.0,
        le=1.0,
    )
    tick_interval_seconds: float = Field(
        3600.0,
        description="Interval in seconds between strategy ticks.",
        example=3600.0,
        gt=0,
    )

    @validator("confidence_threshold")
    def _validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    @validator("tick_interval_seconds")
    def _validate_tick_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("tick_interval_seconds must be positive")
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

    # Default values; may be overridden by validated params at runtime.
    confidence_threshold = 0.65
    tick_interval_seconds = 3600.0

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
            Supported keys are ``confidence_threshold`` and ``tick_interval_seconds``.
        """
        params = params or {}
        # Validate and apply strategy‑specific parameters.
        validated_params = StrategyParams(**params)

        # Override instance attributes with validated values.
        self.confidence_threshold = validated_params.confidence_threshold
        self.tick_interval_seconds = validated_params.tick_interval_seconds

        # Preserve original behavior for the underlying momentum strategy.
        super().__init__(params)
        self._base = MomentumStrategy(params)

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
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                return None

            return self._apply_ml_filter(base_signal, ml_result)
        except Exception as e:  # pragma: no cover
            logger.exception("ML inference failed for %s: %s", symbol, e)
            return None

    def _apply_ml_filter(self, base_signal: Signal, ml_result: Dict[str, Any]) -> Optional[Signal]:
        """Adjust the base signal if the ML prediction agrees.

        Parameters
        ----------
        base_signal : Signal
            Signal produced by the underlying momentum strategy.
        ml_result : dict
            Result from the ML inference service containing ``prediction`` and
            ``confidence`` keys.

        Returns
        -------
        Signal | None
            Updated signal if directions match and confidence meets the threshold,
            otherwise ``None``.
        """
        prediction = ml_result["prediction"]
        ml_conf = ml_result["confidence"]

        side_match = (
            (prediction == "up" and base_signal.side == "buy")
            or (prediction == "down" and base_signal.side == "sell")
        )
        if not side_match:
            return None

        # Combine confidences, respecting the configured maximum.
        combined_confidence = min(0.95, (base_signal.confidence + ml_conf) / 2)
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_conf
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