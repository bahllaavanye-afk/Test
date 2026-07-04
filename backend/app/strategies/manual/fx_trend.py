"""FX EMA-crossover trend following (Forex desk)."""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class FXTrendStrategy(AbstractStrategy):
    name = "fx_trend"
    display_name = "FX EMA Trend"
    market_type = "forex"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0

    DEFAULT_PARAMS = {"fast": 20, "slow": 50}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.fast = eff["fast"]
        self.slow = eff["slow"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a signal based on EMA crossovers.

        Handles edge cases such as None inputs, empty data, insufficient rows,
        and off‑by‑one indexing errors.
        """
        # Guard against None or non‑DataFrame inputs
        if data is None or not isinstance(data, pd.DataFrame):
            return None

        # Guard against missing or empty 'close' column
        if "close" not in data.columns or data.empty:
            return None

        # Ensure enough rows for EMA calculation plus a safety buffer
        if len(data) < max(self.fast, self.slow) + 2:
            return None

        close = data["close"]
        fast = ta.ema(close, length=self.fast)
        slow = ta.ema(close, length=self.slow)

        # EMA functions may return None or series shorter than requested
        if fast is None or slow is None or len(fast) < 2 or len(slow) < 2:
            return None

        # Use the last two points for crossover detection
        f, s = fast.iloc[-1], slow.iloc[-1]
        fp, sp = fast.iloc[-2], slow.iloc[-2]

        # Validate that none of the values are NaN
        if any(pd.isna(x) for x in (f, s, fp, sp)):
            return None

        # Detect a fresh golden cross
        if f > s and fp <= sp:
            # Confidence scales with the relative EMA distance, capped at 0.80
            conf = min(0.80, 0.55 + abs(f - s) / max(s, 1e-8) * 5)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=conf,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "fast_ema": round(float(f), 5),
                    "slow_ema": round(float(s), 5),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Create entry and exit signals for backtesting.

        Handles edge cases such as None inputs, missing columns, empty frames,
        and insufficient data for EMA calculations.
        """
        # Guard against invalid inputs
        if df is None or not isinstance(df, pd.DataFrame):
            empty = pd.Series(False, index=pd.Index([]))
            return BacktestSignals(entries=empty, exits=empty)

        if "close" not in df.columns or df.empty:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Ensure enough rows for EMA calculations
        if len(df) < max(self.fast, self.slow) + 2:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        close = df["close"]
        fast = ta.ema(close, length=self.fast)
        slow = ta.ema(close, length=self.slow)

        # If EMA generation fails, return empty signals aligned with df
        if fast is None or slow is None:
            empty = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty, exits=empty)

        # Shift to align with closed bars; handle potential short series
        f = fast.shift(1)
        s = slow.shift(1)
        fp = fast.shift(2)
        sp = slow.shift(2)

        # Build boolean series, filling NaNs with False to avoid errors
        entries = (f > s) & (fp <= sp)
        exits = (f < s) & (fp >= sp)

        entries = entries.fillna(False)
        exits = exits.fillna(False)

        # Align output series with the original dataframe index
        if entries.index.equals(df.index) is False:
            entries = entries.reindex(df.index, fill_value=False)
        if exits.index.equals(df.index) is False:
            exits = exits.reindex(df.index, fill_value=False)

        return BacktestSignals(entries=entries, exits=exits)