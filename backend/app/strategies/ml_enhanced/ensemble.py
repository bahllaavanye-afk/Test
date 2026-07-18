"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


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
        """
        try:
            ml_result = await self._fetch_ml_result(data, symbol)
            if not self._is_valid_ml_result(ml_result):
                return None

            if not self._has_required_columns(data):
                return None

            sma, median_vol = self._compute_confirmation_metrics(data)
            if sma is None or median_vol is None:
                return None

            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            if not self._passes_directional_check(ml_result["prediction"], latest_close, sma):
                return None

            if not self._passes_volume_check(latest_vol, median_vol):
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

    async def _fetch_ml_result(self, data: pd.DataFrame, symbol: str) -> dict | None:
        """Retrieve ML prediction for the given symbol."""
        inference = get_inference_service()
        return await inference.predict(data, symbol)

    def _is_valid_ml_result(self, ml_result: dict | None) -> bool:
        """Validate ML result presence, directionality, and confidence."""
        if not ml_result or ml_result.get("prediction") == "neutral":
            return False
        if ml_result.get("confidence", 0) < self.confidence_threshold:
            return False
        return True

    def _has_required_columns(self, data: pd.DataFrame) -> bool:
        """Ensure price and volume columns are available."""
        return {"close", "volume"}.issubset(data.columns)

    def _compute_confirmation_metrics(self, data: pd.DataFrame) -> tuple[float | None, float | None]:
        """Calculate SMA and median volume over the recent window."""
        recent = data.tail(self.sma_window)
        if recent.empty:
            return None, None
        sma = recent["close"].mean()
        median_vol = recent["volume"].median()
        return sma, median_vol

    def _passes_directional_check(self, prediction: str, latest_close: float, sma: float) -> bool:
        """Confirm price is on the correct side of SMA for the prediction."""
        if prediction == "up":
            return latest_close > sma
        # prediction == "down"
        return latest_close < sma

    def _passes_volume_check(self, latest_vol: float, median_vol: float) -> bool:
        """Confirm volume is above the median of the recent window."""
        return latest_vol >= median_vol

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
        required_cols = {"close", "volume", "ml_prediction", "ml_confidence"}
        if not required_cols.issubset(df.columns):
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()

        is_up = df["ml_prediction"] == "up"
        is_down = df["ml_prediction"] == "down"
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        price_above_sma = df["close"] > sma
        price_below_sma = df["close"] < sma
        vol_ok = df["volume"] >= median_vol

        long_entry = is_up & conf_ok & price_above_sma & vol_ok
        short_entry = is_down & conf_ok & price_below_sma & vol_ok

        entries = long_entry | short_entry

        exit_long = (~price_above_sma) | (~vol_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_below_sma) | (~vol_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)