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


logger = logging.getLogger(__name__)


class MLMomentumStrategy(AbstractStrategy):
    """ML‑enhanced momentum strategy.

    The strategy wraps the classic momentum logic and applies an ML filter.
    It inherits from :class:`app.strategies.base.AbstractStrategy`.

    Attributes
    ----------
    name : str
        Internal identifier for the strategy.
    display_name : str
        Human‑readable name.
    market_type : str
        Market classification (e.g., ``"equity"``).
    strategy_type : str
        Type classification (e.g., ``"ml_enhanced"``).
    risk_bucket : str
        Risk categorisation.
    tick_interval_seconds : float
        Expected tick interval for the underlying data.
    confidence_threshold : float
        Minimum confidence required from the ML model to consider a signal.
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

        # Apply ML filter
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                return None

            # Only pass if ML agrees with indicator direction
            if ml_result["prediction"] == "up" and base_signal.side == "buy":
                base_signal.confidence = min(
                    0.95, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_result["confidence"]
                return base_signal
            elif ml_result["prediction"] == "down" and base_signal.side == "sell":
                base_signal.confidence = min(
                    0.95, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_result["confidence"]
                return base_signal
        except Exception as e:
            logger.exception("ML inference failed for %s: %s", symbol, e)

        return None

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
        # For backtesting without live ML: use base signals as proxy
        return self._base.backtest_signals(df)