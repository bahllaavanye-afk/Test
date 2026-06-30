"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import logging
import time
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service

_logger = logging.getLogger(__name__)


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML
    sma_window = 20  # simple moving average window for confirmation

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a trading signal based on the ML inference combined with
        price‑based confirmation filters.

        Entry Conditions
        ----------------
        1. ML model predicts a directional move (up/down) with confidence >= threshold.
        2. Current close price is above the SMA for a long signal, or below the SMA for a short.
        3. Volume is above the median of the recent window (default 20 periods).

        Exit Conditions
        ----------------
        A signal is not emitted if any of the above conditions fail, which the
        back‑testing engine interprets as an exit for the active position.
        """
        start_time = time.perf_counter()
        signal: Signal | None = None
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

            signal = Signal(
                symbol=symbol,
                side="buy" if ml_result["prediction"] == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
            return signal
        finally:
            exec_time = time.perf_counter() - start_time
            signal_count = 1 if signal else 0
            # P&L is not available in real‑time analysis; set to None.
            _logger.info(
                "EnsembleStrategy analyze completed",
                extra={
                    "signal_count": signal_count,
                    "execution_time_seconds": exec_time,
                    "pnl": None,
                },
            )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing.

        Expected DataFrame columns:
        - 'close': price series
        - 'volume': volume series
        - 'ml_prediction': string ("up", "down", "neutral")
        - 'ml_confidence': float (0‑1)

        The method mirrors the runtime `analyze` logic but operates row‑wise.
        """
        start_time = time.perf_counter()
        required_cols = {"close", "volume", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            # If required columns are missing, return empty signals to avoid crashes.
            empty = pd.Series(False, index=df.index)
            exec_time = time.perf_counter() - start_time
            _logger.info(
                "EnsembleStrategy backtest_signals completed",
                extra={
                    "signal_count": 0,
                    "execution_time_seconds": exec_time,
                    "pnl": None,
                },
            )
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

        exec_time = time.perf_counter() - start_time
        signal_count = int(entries.sum())
        _logger.info(
            "EnsembleStrategy backtest_signals completed",
            extra={
                "signal_count": signal_count,
                "execution_time_seconds": exec_time,
                "pnl": None,
            },
        )
        return BacktestSignals(entries=entries, exits=exits)