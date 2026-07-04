"""Commodity z-score mean reversion (Commodities desk).

The desk's two existing strategies (``commodity_momentum``, ``commodity_trend``)
are both long-only trend-followers — the *same* factor. They sit flat or get
whipsawed in the range-bound, mean-reverting regimes that dominate commodity
tapes between trends. This adds the missing complementary edge: a **two-sided**
counter-trend fade.

Edge: commodities oscillate around a slow-moving fair value (storage/convenience
yield anchor). When price stretches far from its rolling mean — measured as a
z-score of close vs a rolling mean/std — it tends to snap back. Go long when the
z-score is deeply negative (oversold), short when deeply positive (overbought),
and flatten as it reverts toward the mean.

Causality: the live ``analyze`` reads the latest closed bar; ``backtest_signals``
shifts the z-score by one bar so the decision at bar t uses only data through t-1.
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CommodityReversionStrategy(AbstractStrategy):
    name = "commodity_reversion"
    display_name = "Commodity Z-Score Mean Reversion"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    # window: rolling lookback for the mean/std; entry_z: how stretched before we
    # fade; exit_z: revert-to-mean band where we flatten (|z| <= exit_z).
    DEFAULT_PARAMS = {"window": 20, "entry_z": 2.0, "exit_z": 0.5}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.window = int(eff["window"])
        self.entry_z = float(eff["entry_z"])
        self.exit_z = float(eff["exit_z"])

    def _zscore(self, close: pd.Series) -> pd.Series:
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        return (close - mean) / std.replace(0, pd.NA)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if "close" not in data.columns or len(data) < self.window + 5:
            return None
        z = self._zscore(data["close"]).iloc[-1]
        if pd.isna(z):
            return None
        if z <= -self.entry_z:  # stretched below the mean → fade up (long)
            conf = min(0.85, 0.55 + (abs(z) - self.entry_z) * 0.1)
            return Signal(symbol=symbol, side="buy", confidence=conf,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"zscore": round(float(z), 2), "window": self.window})
        if z >= self.entry_z:   # stretched above the mean → fade down (short)
            conf = min(0.85, 0.55 + (abs(z) - self.entry_z) * 0.1)
            return Signal(symbol=symbol, side="sell", confidence=conf,
                          strategy_name=self.name, strategy_type=self.strategy_type,
                          risk_bucket=self.risk_bucket,
                          metadata={"zscore": round(float(z), 2), "window": self.window})
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        z = self._zscore(df["close"]).shift(1)  # decide today from yesterday's z-score
        entries = z <= -self.entry_z            # long when oversold
        exits = z >= -self.exit_z               # flatten as it reverts toward the mean
        short_entries = z >= self.entry_z       # short when overbought
        short_exits = z <= self.exit_z          # cover as it reverts toward the mean
        return BacktestSignals(
            entries=entries.fillna(False), exits=exits.fillna(False),
            short_entries=short_entries.fillna(False), short_exits=short_exits.fillna(False),
        )
