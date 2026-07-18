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

# Constants
TICK_INTERVAL_SECONDS = 86400.0
DEFAULT_WINDOW = 20
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.5
CONFIDENCE_CAP = 0.85
CONFIDENCE_BASE = 0.55
CONFIDENCE_MULT = 0.1
MIN_DATA_BUFFER = 5
SIDE_BUY = "buy"
SIDE_SELL = "sell"
COLUMN_CLOSE = "close"
METADATA_ZSCORE = "zscore"
METADATA_WINDOW = "window"


class CommodityReversionStrategy(AbstractStrategy):
    name = "commodity_reversion"
    display_name = "Commodity Z-Score Mean Reversion"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = TICK_INTERVAL_SECONDS

    # window: rolling lookback for the mean/std; entry_z: how stretched before we
    # fade; exit_z: revert-to-mean band where we flatten (|z| <= exit_z).
    DEFAULT_PARAMS = {
        "window": DEFAULT_WINDOW,
        "entry_z": DEFAULT_ENTRY_Z,
        "exit_z": DEFAULT_EXIT_Z,
    }

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
        if COLUMN_CLOSE not in data.columns or len(data) < self.window + MIN_DATA_BUFFER:
            return None
        z = self._zscore(data[COLUMN_CLOSE]).iloc[-1]
        if pd.isna(z):
            return None
        if z <= -self.entry_z:  # stretched below the mean → fade up (long)
            conf = min(CONFIDENCE_CAP, CONFIDENCE_BASE + (abs(z) - self.entry_z) * CONFIDENCE_MULT)
            return Signal(
                symbol=symbol,
                side=SIDE_BUY,
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={METADATA_ZSCORE: round(float(z), 2), METADATA_WINDOW: self.window},
            )
        if z >= self.entry_z:   # stretched above the mean → fade down (short)
            conf = min(CONFIDENCE_CAP, CONFIDENCE_BASE + (abs(z) - self.entry_z) * CONFIDENCE_MULT)
            return Signal(
                symbol=symbol,
                side=SIDE_SELL,
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={METADATA_ZSCORE: round(float(z), 2), METADATA_WINDOW: self.window},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        z = self._zscore(df[COLUMN_CLOSE]).shift(1)  # decide today from yesterday's z-score
        entries = z <= -self.entry_z            # long when oversold
        exits = z >= -self.exit_z               # flatten as it reverts toward the mean
        short_entries = z >= self.entry_z       # short when overbought
        short_exits = z <= self.exit_z          # cover as it reverts toward the mean
        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )