"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters.

This module defines :class:`EnsembleStrategy`, an implementation of
:class:`~app.strategies.base.AbstractStrategy` that aggregates predictions from
the ML inference service (LSTM, XGBoost, Lorentzian) and applies price/volume
confirmation filters before emitting a :class:`~app.strategies.base.Signal`.
"""

import pandas as pd
from typing import Optional, Dict, Any

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class EnsembleStrategy(AbstractStrategy):
    """Concrete strategy that combines ML predictions with SMA and volume filters.

    Attributes
    ----------
    name : str
        Internal identifier for the strategy.
    display_name : str
        Human‑readable name shown in UI / logs.
    market_type : str
        Asset class the strategy targets.
    strategy_type : str
        Category of the strategy (ml_enhanced).
    risk_bucket : str
        Risk classification used for allocation.
    tick_interval_seconds : float
        Minimum time between consecutive evaluations.
    confidence_threshold : float
        Minimum confidence required from the ML model to consider a signal.
    sma_window : int
        Look‑back window for the simple moving average used as a price filter.
    """

    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML
    sma_window = 20  # simple moving average window for confirmation

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a live trading signal.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and volume data required for the SMA and volume
            confirmation filters. Must contain ``close`` and ``volume`` columns.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Optional[Signal]
            A populated :class:`Signal` when all entry conditions are met,
            otherwise ``None`` indicating no actionable signal.
        """
        try:
            inference = get_inference_service()
            ml_result: Dict[str, Any] = await inference.predict(data, symbol)

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

        The logic mirrors :meth:`analyze` but operates on a DataFrame that already
        contains the ML inference results. It returns boolean Series indicating
        where entries and exits would occur.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the columns ``close``, ``volume``,
            ``ml_prediction`` and ``ml_confidence``.

        Returns
        -------
        BacktestSignals
            Container with ``entries`` and ``exits`` boolean Series aligned with
            the input index.
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