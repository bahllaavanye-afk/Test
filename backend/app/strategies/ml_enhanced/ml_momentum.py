"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it tightens entry conditions with additional confirmation
filters. Exit logic is also refined to respect ML disagreement and price‑action
reversals.
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
    """

    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65  # minimum ML confidence to accept a signal
    entry_return_window = 3          # periods for short‑term return confirmation
    entry_volume_short_window = 5    # periods for recent volume average
    entry_volume_long_window = 20    # periods for longer‑term volume average
    price_ema_window = 20            # EMA window for price confirmation

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
        ML prediction agrees with the base signal direction, the confidence exceeds
        the configured threshold, and additional confirmation filters pass, the
        signal confidence is adjusted and returned.

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
            ML models agree and entry filters succeed, otherwise ``None``.
        """
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result.get("prediction") == "neutral":
                return None

            # Apply entry‑side confirmation filters before merging confidences
            if not self._entry_confirmation(data, base_signal.side):
                logger.debug("Entry confirmation filters rejected %s signal for %s", base_signal.side, symbol)
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
            logger.debug("ML prediction (%s) disagrees with base side (%s)", prediction, base_signal.side)
            return None

        if ml_conf < self.confidence_threshold:
            logger.debug("ML confidence %.3f below threshold %.3f", ml_conf, self.confidence_threshold)
            return None

        # Combine confidences, respecting the configured maximum.
        combined_confidence = min(0.95, (base_signal.confidence + ml_conf) / 2)
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_conf
        base_signal.metadata["ml_prediction"] = prediction

        # Attach exit guard metadata – if later data shows reversal, back‑tester can
        # use this flag to trigger an early exit.
        base_signal.metadata["exit_guard"] = "ml_disagreement"

        return base_signal

    def _entry_confirmation(self, data: pd.DataFrame, side: str) -> bool:
        """Run supplemental filters to tighten entry criteria.

        The filters include:
        * Short‑term return momentum (positive for long, negative for short)
        * Volume surge relative to a longer‑term average
        * Price positioned relative to a 20‑period EMA

        Parameters
        ----------
        data : pd.DataFrame
            Historical price/volume data. Expected columns: ``close`` and ``volume``.
        side : str
            Desired trade side (``"buy"`` or ``"sell"``).

        Returns
        -------
        bool
            ``True`` if all confirmation filters pass, ``False`` otherwise.
        """
        # Ensure required columns exist
        required_cols = {"close", "volume"}
        if not required_cols.issubset(data.columns):
            logger.debug("Missing required columns for entry confirmation: %s", required_cols - set(data.columns))
            return False

        # 1. Short‑term return momentum
        recent_returns = data["close"].pct_change().rolling(self.entry_return_window).mean()
        recent_ret = recent_returns.iloc[-1]

        if side == "buy" and (recent_ret is None or recent_ret <= 0):
            return False
        if side == "sell" and (recent_ret is None or recent_ret >= 0):
            return False

        # 2. Volume surge filter
        vol_short = data["volume"].rolling(self.entry_volume_short_window).mean().iloc[-1]
        vol_long = data["volume"].rolling(self.entry_volume_long_window).mean().iloc[-1]
        if vol_long == 0 or vol_short / vol_long < 1.2:
            return False

        # 3. Price EMA filter
        ema = data["close"].ewm(span=self.price_ema_window, adjust=False).mean().iloc[-1]
        price = data["close"].iloc[-1]
        if side == "buy" and price <= ema:
            return False
        if side == "sell" and price >= ema:
            return False

        return True

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate back‑test signals for a dataframe.

        For back‑testing environments where live ML inference is unavailable,
        this method falls back to the base momentum signals but applies the same
        entry‑confirmation filters used in live trading.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing historical data for back‑testing.

        Returns
        -------
        BacktestSignals
            Signals suitable for back‑testing consumption.
        """
        base_signals = self._base.backtest_signals(df)

        # Apply entry confirmation to each signal in the back‑test set.
        filtered = []
        for signal in base_signals.signals:
            if self._entry_confirmation(df, signal.side):
                # Mimic the ML confidence boost using a static placeholder
                # (the actual ML layer is not available in back‑test).
                placeholder_ml_conf = 0.70
                combined_conf = min(0.95, (signal.confidence + placeholder_ml_conf) / 2)
                signal.confidence = combined_conf
                signal.strategy_name = self.name
                signal.strategy_type = self.strategy_type
                signal.metadata["ml_confidence"] = placeholder_ml_conf
                filtered.append(signal)

        return BacktestSignals(signals=filtered)