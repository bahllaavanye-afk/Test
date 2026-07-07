"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it adjusts the confidence accordingly.
"""

import logging
import time
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

        # Monitoring metrics
        self._signal_count: int = 0
        self._total_pnl: float = 0.0  # Placeholder; actual P&L is calculated downstream

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
        start_time = time.time()
        signal_generated = False

        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "MLMomentum analysis completed",
                extra={
                    "symbol": symbol,
                    "signal_generated": False,
                    "execution_time_ms": round(duration_ms, 2),
                    "total_signal_count": self._signal_count,
                    "pnl": None,
                },
            )
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result["prediction"] == "neutral":
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    "MLMomentum analysis completed",
                    extra={
                        "symbol": symbol,
                        "signal_generated": False,
                        "execution_time_ms": round(duration_ms, 2),
                        "total_signal_count": self._signal_count,
                        "pnl": None,
                    },
                )
                return None

            final_signal = self._apply_ml_filter(base_signal, ml_result)
            if final_signal is not None:
                self._signal_count += 1
                signal_generated = True

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "MLMomentum analysis completed",
                extra={
                    "symbol": symbol,
                    "signal_generated": signal_generated,
                    "execution_time_ms": round(duration_ms, 2),
                    "total_signal_count": self._signal_count,
                    "pnl": None,
                },
            )
            return final_signal
        except Exception as e:  # pragma: no cover
            duration_ms = (time.time() - start_time) * 1000
            logger.exception("ML inference failed for %s: %s", symbol, e)
            logger.info(
                "MLMomentum analysis completed with error",
                extra={
                    "symbol": symbol,
                    "signal_generated": False,
                    "execution_time_ms": round(duration_ms, 2),
                    "total_signal_count": self._signal_count,
                    "pnl": None,
                },
            )
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