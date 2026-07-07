"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters.

This module defines :class:`EnsembleStrategy`, a concrete implementation of
:class:`app.strategies.base.AbstractStrategy`.  The strategy uses a downstream
ML inference service (LSTM, XGBoost, Lorentzian) to generate a directional
prediction and applies price‑ and volume‑based confirmation filters before
emitting a :class:`app.strategies.base.Signal`.

The strategy is intended for equity markets and operates on 5‑minute bars
(`tick_interval_seconds = 300`).  It is used both in live trading (via the
asynchronous :meth:`EnsembleStrategy.analyze` method) and in back‑testing
(via :meth:`EnsembleStrategy.backtest_signals`).
"""

from __future__ import annotations

import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class EnsembleStrategy(AbstractStrategy):
    """Concrete strategy that combines ML inference with SMA and volume filters."""

    name: str = "ensemble"
    display_name: str = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type: str = "equity"
    strategy_type: str = "ml_enhanced"
    risk_bucket: str = "directional"
    tick_interval_seconds: float = 300.0
    confidence_threshold: float = 0.70  # higher bar for pure ML
    sma_window: int = 20  # simple moving average window for confirmation

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a live trading signal.

        Parameters
        ----------
        data : pandas.DataFrame
            Historical bar data for ``symbol``.  Must contain at least the
            ``close`` and ``volume`` columns.
        symbol : str
            Ticker symbol to which the signal applies.

        Returns
        -------
        Signal | None
            A :class:`Signal` object when all entry conditions are satisfied,
            otherwise ``None`` (interpreted by the back‑testing engine as an exit).

        Notes
        -----
        The method performs the following steps:

        1. Retrieves an inference service and obtains a prediction.
        2. Checks that the prediction is not ``neutral`` and that its confidence
           meets ``confidence_threshold``.
        3. Computes a simple moving average (SMA) and median volume over the
           most recent ``sma_window`` periods.
        4. Confirms that the latest price is above the SMA for a long signal
           (or below for a short) and that the latest volume exceeds the median.
        """
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Basic ML validation
            if not ml_result or ml_result.get("prediction") == "neutral":
                return None
            if ml_result.get("confidence", 0) < self.confidence_threshold:
                return None

            # Ensure we have price and volume data for confirmation
            if "close" not in data.columns or "volume" not in data.columns:
                return None

            # Compute SMA and median volume on the latest slice
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None
            sma = recent["close"].mean()
            median_vol = recent["volume"].median()
            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            # Directional confirmation
            if ml_result["prediction"] == "up":
                if latest_close <= sma:
                    return None
            else:  # prediction == "down"
                if latest_close >= sma:
                    return None

            # Volume confirmation
            if latest_vol < median_vol:
                return None

            return Signal(
                symbol=symbol,
                side="buy" if ml_result["prediction"] == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
        except Exception:
            # In production we would log the exception; for now we silently ignore.
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate entry and exit signals for back‑testing.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing historical bars for a single symbol.  Required
            columns are ``close``, ``volume``, ``ml_prediction`` and
            ``ml_confidence``.

        Returns
        -------
        BacktestSignals
            A container with two boolean ``pandas.Series``: ``entries`` and
            ``exits``.  Each series aligns with ``df``'s index.

        Notes
        -----
        The logic mirrors :meth:`analyze` but operates row‑wise using rolling
        calculations for SMA and median volume.
        """
        required_cols = {"close", "volume", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            # If required columns are missing, return empty signals to avoid crashes.
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Compute rolling SMA and median volume
        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()

        # Conditions for a valid entry
        is_up = df["ml_prediction"] == "up"
        is_down = df["ml_prediction"] == "down"
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        price_above_sma = df["close"] > sma
        price_below_sma = df["close"] < sma
        vol_ok = df["volume"] >= median_vol

        long_entry = is_up & conf_ok & price_above_sma & vol_ok
        short_entry = is_down & conf_ok & price_below_sma & vol_ok

        entries = long_entry | short_entry

        # Exit when any of the entry conditions become false for the current side.
        # For simplicity we treat the opposite side as an exit signal.
        exit_long = (~price_above_sma) | (~vol_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_below_sma) | (~vol_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        # Align boolean Series with BacktestSignals expectations
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)