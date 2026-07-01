"""
Bollinger Band Mean Reversion Strategy.

This strategy generates buy signals when the price touches the lower Bollinger Band
and the RSI is oversold, and sell signals when the price touches the upper
Bollinger Band and the RSI is overbought.  The target price for both entry
directions is the middle Bollinger Band.

The implementation provides both a live `analyze` method (asynchronous) and a
`backtest_signals` method for historical evaluation.
"""

import pandas as pd
from app.ml.features import pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class MeanReversionStrategy(AbstractStrategy):
    """Bollinger Band mean‑reversion strategy."""

    name = "mean_reversion"
    display_name = "Bollinger Band Mean Reversion"
    market_type = "equity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0

    DEFAULT_PARAMS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }

    def __init__(self, params: dict | None = None):
        """
        Initialise the strategy with optional parameter overrides.

        Parameters
        ----------
        params : dict | None
            Dictionary containing any of the keys in ``DEFAULT_PARAMS`` to
            override the defaults.
        """
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.bb_period = effective["bb_period"]
        self.bb_std = effective["bb_std"]
        self.rsi_period = effective["rsi_period"]
        self.rsi_oversold = effective["rsi_oversold"]
        self.rsi_overbought = effective["rsi_overbought"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyse the latest market data and emit a signal if conditions are met.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with at least a ``close`` column.
        symbol : str
            Ticker symbol for which the signal is generated.

        Returns
        -------
        Signal | None
            A populated ``Signal`` instance when entry criteria are satisfied,
            otherwise ``None``.
        """
        if "close" not in data.columns or len(data) < self.bb_period + 5:
            return None

        close = data["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            return None

        # Extract the latest Bollinger Band values
        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].iloc[-1]
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].iloc[-1]
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].iloc[-1]
        price = close.iloc[-1]
        rsi_val = rsi.iloc[-1]

        # Long entry condition
        if price <= lower and rsi_val < self.rsi_oversold:
            pct_below = (lower - price) / lower
            confidence = min(0.88, 0.55 + pct_below * 5)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=mid,
                metadata={"rsi": round(rsi_val, 2), "bb_position": "lower"},
            )

        # Short entry condition
        if price >= upper and rsi_val > self.rsi_overbought:
            pct_above = (price - upper) / upper
            confidence = min(0.88, 0.55 + pct_above * 5)
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=mid,
                metadata={"rsi": round(rsi_val, 2), "bb_position": "upper"},
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing.

        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV data containing a ``close`` column.

        Returns
        -------
        BacktestSignals
            Named tuple with boolean series for entries, exits, short entries,
            and short exits.
        """
        close = df["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Shift by one period to avoid look‑ahead bias
        lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].shift(1)
        upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].shift(1)
        mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].shift(1)
        rsi_s = rsi.shift(1)
        close_s = close.shift(1)

        entries = (close_s <= lower) & (rsi_s < self.rsi_oversold)
        exits = close_s >= mid
        short_entries = (close_s >= upper) & (rsi_s > self.rsi_overbought)
        short_exits = close_s <= mid

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )