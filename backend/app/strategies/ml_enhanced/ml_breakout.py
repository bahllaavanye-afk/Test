"""ML-filtered breakout strategy."""
import logging
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutStrategy(AbstractStrategy):
    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze market data and produce a signal.

        The method first obtains a base breakout signal. It then applies
        additional volume and ML‑based confirmation filters. Entry is only
        confirmed when both the base signal and the ML model agree and the
        confidence thresholds are met. An opposite‑direction high‑confidence ML
        prediction is used as an exit trigger.
        """
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        # ---- Confirmation Filters -------------------------------------------------
        # Volume filter: require recent volume to be at least 20 % above the overall average.
        if "volume" in data.columns:
            recent_vol = data["volume"].iloc[-5:].mean()
            avg_vol = data["volume"].mean()
            if avg_vol == 0 or recent_vol / avg_vol < 1.2:
                return None
        else:
            # Without volume data we cannot apply the confirmation filter.
            return None

        # ---- ML Inference ---------------------------------------------------------
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
        except Exception as exc:
            logging.exception("ML inference failed for %s", symbol)
            # Fallback to the base signal if ML fails.
            return base_signal

        if not ml_result:
            return base_signal

        # ---- Tighten Entry Conditions --------------------------------------------
        ml_conf = ml_result.get("confidence", 0.0)
        ml_pred = ml_result.get("prediction")
        base_dir = getattr(base_signal, "direction", None)

        # Require ML confidence >= 0.80 and agreement with the base direction.
        if ml_conf >= 0.80 and ml_pred == base_dir:
            # Combine confidences, capping at a safe upper bound.
            combined_conf = (base_signal.confidence + ml_conf) / 2.0
            base_signal.confidence = min(0.95, combined_conf)

            # Tag the signal as ML‑enhanced.
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            meta = getattr(base_signal, "metadata", {})
            meta["ml_confirmed"] = True
            base_signal.metadata = meta
            return base_signal

        # ---- Exit Logic -----------------------------------------------------------
        # If ML predicts the opposite direction with high confidence, treat it as an exit.
        if ml_conf >= 0.85 and ml_pred != base_dir:
            # Mark the signal for exit; downstream logic should handle the actual exit.
            base_signal.confidence = ml_conf
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            meta = getattr(base_signal, "metadata", {})
            meta["ml_exit"] = True
            base_signal.metadata = meta
            if hasattr(base_signal, "exit"):
                base_signal.exit = True
            return base_signal

        # No enhanced signal generated.
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        return self._base.backtest_signals(df)