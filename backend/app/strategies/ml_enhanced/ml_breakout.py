"""ML-filtered breakout strategy.

This module defines the :class:`MLBreakoutStrategy`, which enhances a classic
breakout strategy with a machine‑learning filter. The base breakout logic is
provided by :class:`~app.strategies.manual.breakout.BreakoutStrategy`. After the
base signal is generated, an ML inference service is queried; if the model
predicts an upward move with sufficient confidence, the signal's confidence is
adjusted and the strategy metadata is updated.
"""

import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutStrategy(AbstractStrategy):
    """ML‑enhanced breakout strategy.

    Attributes
    ----------
    name : str
        Internal identifier for the strategy.
    display_name : str
        Human‑readable name shown in UI or logs.
    market_type : str
        Market segment the strategy targets (e.g., ``"equity"``).
    strategy_type : str
        Category of the strategy, here ``"ml_enhanced"``.
    risk_bucket : str
        Risk classification used for portfolio allocation.
    tick_interval_seconds : float
        Minimum interval between ticks the strategy processes.
    """

    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: dict | None = None) -> None:
        """Create a new instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration dictionary passed to the underlying
            :class:`BreakoutStrategy`. If ``None``, defaults are used.
        """
        super().__init__(params)
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a trading signal using the base breakout and ML filter.

        The method first obtains a signal from the underlying breakout strategy.
        If a base signal exists, it queries the ML inference service. When the
        model predicts an upward move with confidence greater than ``0.65``,
        the original signal's confidence is blended with the ML confidence and
        capped at ``0.92``. The strategy metadata is also updated.

        Parameters
        ----------
        data : pandas.DataFrame
            Historical price and volume data required by both the breakout and
            ML models.
        symbol : str
            Ticker symbol being evaluated.

        Returns
        -------
        Signal | None
            The adjusted signal if the ML filter approves it, the original base
            signal on inference failure, or ``None`` if no base signal was
            generated.
        """
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result["confidence"] > 0.65
                and ml_result["prediction"] == "up"
            ):
                base_signal.confidence = min(
                    0.92, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Delegate backtesting to the underlying breakout strategy.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing historical market data for backtesting.

        Returns
        -------
        BacktestSignals
            Signals produced by the base breakout strategy for the given data.
        """
        return self._base.backtest_signals(df)