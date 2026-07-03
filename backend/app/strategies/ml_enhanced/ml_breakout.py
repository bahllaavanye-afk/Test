"""ML-filtered breakout strategy."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutStrategy(AbstractStrategy):
    """A breakout strategy enhanced with ML inference.

    The strategy first runs the classic breakout logic and, if a base
    signal is generated, it queries the ML inference service.  When the
    ML model is confident enough (confidence > 0.65) and predicts an
    upward move, the base signal confidence is adjusted and the signal
    is returned.  Edge‑case handling is added for ``None`` inputs,
    empty data frames, missing keys, and off‑by‑one confidence
    calculations.
    """

    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        """Generate a signal for a single symbol.

        Returns ``None`` if the input data is missing/empty, the symbol is
        falsy, or the base breakout strategy does not generate a signal.
        """
        # Guard against None or empty inputs
        if data is None or data.empty or not symbol:
            return None

        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Ensure ml_result is a dict with expected keys
            if not isinstance(ml_result, dict):
                return base_signal

            confidence = ml_result.get("confidence")
            prediction = ml_result.get("prediction")

            if confidence is None or prediction is None:
                return base_signal

            if confidence > 0.65 and prediction == "up":
                # Prevent off‑by‑one errors by capping at 0.92 after averaging
                avg_conf = (base_signal.confidence + confidence) / 2
                base_signal.confidence = min(0.92, avg_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            # In case of any failure, fall back to the base signal
            return base_signal

        return None

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        """Return back‑test signals for a DataFrame.

        If ``df`` is ``None`` or empty, an empty ``BacktestSignals`` object
        is returned to avoid downstream errors.
        """
        if df is None or df.empty:
            # Assuming BacktestSignals can be instantiated with an empty list
            return BacktestSignals([])
        return self._base.backtest_signals(df)