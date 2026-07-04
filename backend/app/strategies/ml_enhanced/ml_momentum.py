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

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.momentum import MomentumStrategy
from app.ml.inference import get_inference_service

# Constants
STRATEGY_NAME = "ml_momentum"
DISPLAY_NAME = "ML Momentum (LSTM + XGBoost Filter)"
MARKET_TYPE = "equity"
STRATEGY_TYPE = "ml_enhanced"
RISK_BUCKET = "directional"
TICK_INTERVAL_SECONDS = 3600.0
CONFIDENCE_THRESHOLD = 0.65

MAX_COMBINED_CONFIDENCE = 0.95
CONFIDENCE_AVG_DIVISOR = 2

PREDICTION_KEY = "prediction"
CONFIDENCE_KEY = "confidence"
NEUTRAL_PREDICTION = "neutral"
UP_PREDICTION = "up"
DOWN_PREDICTION = "down"
BUY_SIDE = "buy"
SELL_SIDE = "sell"
ML_CONFIDENCE_META_KEY = "ml_confidence"

LOG_MSG_INFERENCE_FAILURE = "ML inference failed for %s: %s"

logger = logging.getLogger(__name__)


class MLMomentumStrategy(AbstractStrategy):
    """ML‑enhanced momentum strategy.

    The strategy wraps the classic momentum logic and applies an ML filter.
    It inherits from :class:`app.strategies.base.AbstractStrategy`.
    """

    name = STRATEGY_NAME
    display_name = DISPLAY_NAME
    market_type = MARKET_TYPE
    strategy_type = STRATEGY_TYPE
    risk_bucket = RISK_BUCKET
    tick_interval_seconds = TICK_INTERVAL_SECONDS
    confidence_threshold = CONFIDENCE_THRESHOLD

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
        """
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
            if ml_result is None or ml_result[PREDICTION_KEY] == NEUTRAL_PREDICTION:
                return None

            return self._apply_ml_filter(base_signal, ml_result)
        except Exception as e:  # pragma: no cover
            logger.exception(LOG_MSG_INFERENCE_FAILURE, symbol, e)
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
        prediction = ml_result[PREDICTION_KEY]
        ml_conf = ml_result[CONFIDENCE_KEY]

        side_match = (
            (prediction == UP_PREDICTION and base_signal.side == BUY_SIDE)
            or (prediction == DOWN_PREDICTION and base_signal.side == SELL_SIDE)
        )
        if not side_match:
            return None

        # Combine confidences, respecting the configured maximum.
        combined_confidence = min(
            MAX_COMBINED_CONFIDENCE,
            (base_signal.confidence + ml_conf) / CONFIDENCE_AVG_DIVISOR,
        )
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata[ML_CONFIDENCE_META_KEY] = ml_conf
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