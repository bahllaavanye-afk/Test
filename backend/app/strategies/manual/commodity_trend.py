"""Commodity SMA-trend breakout (Commodities desk).

Long when a fast SMA is above a slow SMA *and* price clears the recent high —
a trend‑confirmation filter on top of a breakout, suited to commodities' long
directional runs. Exits when price breaks the recent low or the trend weakens.
"""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class CommodityTrendStrategy(AbstractStrategy):
    name = "commodity_trend"
    display_name = "Commodity SMA‑Trend Breakout"
    market_type = "commodity"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86400.0

    DEFAULT_PARAMS = {"fast": 30, "slow": 100, "breakout": 20, "exit_break": 10}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        eff = {**self.DEFAULT_PARAMS, **(params or {})}
        self.fast = eff["fast"]
        self.slow = eff["slow"]
        self.breakout = eff["breakout"]
        self.exit_break = eff["exit_break"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Generate a buy signal if entry criteria are met.

        Entry criteria:
        * Fast SMA > Slow SMA.
        * Fast SMA is rising (current > previous).
        * Current price clears the recent breakout high by at least 0.5%.
        * Current price is also above the fast SMA (adds momentum confirmation).

        Returns a Signal with confidence scaled by the distance above the breakout.
        """
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(data.columns) or len(data) < self.slow + 5:
            return None

        close = data["close"]
        # SMA series (shifted to avoid look‑ahead bias)
        fast_sma_series = close.rolling(self.fast).mean()
        slow_sma_series = close.rolling(self.slow).mean()

        fast = fast_sma_series.iloc[-1]
        slow = slow_sma_series.iloc[-1]
        prev_fast = fast_sma_series.iloc[-2] if len(fast_sma_series) >= 2 else None

        # Recent breakout high (exclude current bar)
        breakout_high = data["high"].rolling(self.breakout).max().shift(1).iloc[-1]

        price = close.iloc[-1]

        # Guard against missing data
        if any(pd.isna(x) for x in (fast, slow, prev_fast, breakout_high)):
            return None

        # Entry filters
        sma_uptrend = fast > slow and prev_fast is not None and fast > prev_fast
        price_above_breakout = price > breakout_high * 1.005  # >=0.5% above breakout high
        price_above_fast = price > fast

        if sma_uptrend and price_above_breakout and price_above_fast:
            # Confidence grows with the gap above the breakout, capped at 0.80
            gap_ratio = (price - breakout_high) / max(breakout_high, 1e-8)
            confidence = min(0.80, 0.56 + gap_ratio * 4)
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata={
                    "breakout_high": round(float(breakout_high), 4),
                    "fast_sma": round(float(fast), 4),
                },
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Back‑test entry and exit signals with tighter filters.

        Entry:
        * Fast SMA > Slow SMA.
        * Fast SMA rising (current > previous).
        * Close > recent breakout high by ≥0.5%.
        * Close > Fast SMA.

        Exit:
        * Close falls below recent low (exit_break window).
        * OR Fast SMA drops below Slow SMA (trend weakening).
        """
        close = df["close"]
        fast_sma = close.rolling(self.fast).mean().shift(1)
        slow_sma = close.rolling(self.slow).mean().shift(1)
        fast_sma_prev = fast_sma.shift(1)

        # Breakout high and exit low (exclude current bar)
        breakout_high = df["high"].rolling(self.breakout).max().shift(2)
        exit_low = df["low"].rolling(self.exit_break).min().shift(2)

        # Entry conditions
        sma_up = (fast_sma > slow_sma) & (fast_sma > fast_sma_prev)
        price_above_breakout = close > breakout_high * 1.005
        price_above_fast = close > fast_sma
        entries = sma_up & price_above_breakout & price_above_fast

        # Exit conditions
        exits = (close < exit_low) | (fast_sma < slow_sma)

        return BacktestSignals(
            entries=entries.fillna(False),
            exits=exits.fillna(False),
        )