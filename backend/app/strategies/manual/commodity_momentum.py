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

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.lookback + 5:
            return None
        close = data["close"]
        mom = close.iloc[-1] / close.iloc[-1 - self.lookback] - 1.0
        if pd.isna(mom) or mom <= 0:
            return None
        conf = min(0.80, 0.55 + min(abs(mom), 0.5) * 0.5)
        return Signal(symbol=symbol, side="buy", confidence=conf,
                      strategy_name=self.name, strategy_type=self.strategy_type,
                      risk_bucket=self.risk_bucket,
                      metadata={"momentum_pct": round(float(mom) * 100, 2),
                                "lookback": self.lookback})

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        close = df["close"]
        # momentum measured on the prior closed bar — no lookahead
        prior = close.shift(1)
        mom = prior / prior.shift(self.lookback) - 1.0
        entries = mom > 0       # long while trailing momentum is positive
        exits = mom < 0         # flat when it turns negative
        return BacktestSignals(entries=entries.fillna(False), exits=exits.fillna(False))
