"""Commodity time-series momentum (Commodities desk).

Classic managed-futures edge: be long while the asset's own trailing return is
positive, flat/short when it turns negative. Computed on lagged prices so the
signal at bar t uses only data through t-1.
"""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta  # noqa: F401  (kept for desk consistency)
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CommodityMomentumStrategy(AbstractStrategy):
    name = "commodity_momentum"
    display_name = "Commodity Time-Series Momentum"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    DEFAULT_PARAMS = {"lookback": 60}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.lookback = eff["lookback"]

    async def analyze(self, data: pd.DataFrame | None, symbol: str) -> Signal | None:
        # Guard against None or malformed inputs
        if data is None or not isinstance(data, pd.DataFrame):
            return None
        # Require the 'close' column and sufficient rows for the lookback calculation
        required_rows = self.lookback + 1
        if "close" not in data.columns or len(data) < required_rows:
            return None
        close = data["close"]
        # Ensure the lookback element exists (handle off‑by‑one edge case)
        if pd.isna(close.iloc[-1]) or pd.isna(close.iloc[-1 - self.lookback]):
            return None
        mom = close.iloc[-1] / close.iloc[-1 - self.lookback] - 1.0
        if pd.isna(mom) or mom <= 0:
            return None
        conf = min(0.80, 0.55 + min(abs(mom), 0.5) * 0.5)
        return Signal(
            symbol=symbol,
            side="buy",
            confidence=conf,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={
                "momentum_pct": round(float(mom) * 100, 2),
                "lookback": self.lookback,
            },
        )

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Guard against None or malformed inputs
        if df is None or not isinstance(df, pd.DataFrame):
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)
        if "close" not in df.columns:
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        close = df["close"]
        if close.empty:
            empty_series = pd.Series(dtype=bool)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        # momentum measured on the prior closed bar — no lookahead
        prior = close.shift(1)
        # Ensure we have enough data for the lookback; if not, fill with NaN which will be treated as False
        mom = prior / prior.shift(self.lookback) - 1.0
        entries = mom > 0  # long while trailing momentum is positive
        exits = mom < 0    # flat when it turns negative
        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )