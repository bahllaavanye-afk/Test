"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters.

This module defines :class:`EnsembleStrategy`, which extends :class:`~app.strategies.base.AbstractStrategy`
to produce trading signals based on a blended ML inference (LSTM, XGBoost, Lorentzian) together with
price‑based confirmation filters (simple moving average and volume median). The strategy is intended for
equity markets and is classified as a directional, ML‑enhanced approach.
"""

import pandas as pd
from typing import Optional, Dict, Any

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


class EnsembleStrategy(AbstractStrategy):
    """Concrete implementation of an ML‑enhanced ensemble trading strategy.

    Attributes
    ----------
    name : str
        Internal identifier used by the back‑testing and execution engines.
    display_name : str
        Human‑readable name shown in UI components.
    market_type : str
        Market classification (e.g., ``"equity"``).
    strategy_type : str
        Category of the strategy; here it is ``"ml_enhanced"``.
    risk_bucket : str
        Risk classification used for position sizing.
    tick_interval_seconds : float
        Minimum interval between successive ticks (5 minutes).
    confidence_threshold : float
        Minimum ML confidence required to consider a prediction actionable.
    sma_window : int
        Look‑back window size for the simple moving average confirmation filter.
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

        The method queries the ML inference service, validates the prediction,
        and applies price/volume confirmation filters.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and volume data for ``symbol``. Must contain ``"close"``
            and ``"volume"`` columns.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Optional[Signal]
            A :class:`~app.strategies.base.Signal` object when all entry criteria are
            satisfied, otherwise ``None`` (interpreted as an exit signal by the
            back‑testing engine).
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
        contains the ML prediction and confidence columns. It returns boolean series
        indicating entry and exit points for each row.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the following columns:
            ``'close'``, ``'volume'``, ``'ml_prediction'`` (``'up'``, ``'down'``,
            ``'neutral'``), and ``'ml_confidence'`` (float between 0 and 1).

        Returns
        -------
        BacktestSignals
            Named tuple with ``entries`` and ``exits`` boolean Series aligned to ``df``.
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