"""
Bollinger Band Mean Reversion Strategy.
Enter when price touches lower/upper band; exit at middle band.
"""
import pandas as pd
import app.ml.features.pandas_ta_compat as ta
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals


class MeanReversionStrategy(AbstractStrategy):
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
        super().__init__(params)
        effective = {**self.DEFAULT_PARAMS, **(params or {})}
        self.bb_period = effective["bb_period"]
        self.bb_std = effective["bb_std"]
        self.rsi_period = effective["rsi_period"]
        self.rsi_oversold = effective["rsi_oversold"]
        self.rsi_overbought = effective["rsi_overbought"]

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        # Guard against None or empty input
        if data is None or data.empty:
            return None
        if "close" not in data.columns:
            return None
        # Ensure enough data points for indicators
        if len(data) < self.bb_period + 5:
            return None

        close = data["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        if bb is None or rsi is None:
            return None

        # Verify required Bollinger columns exist
        lower_col = f"BBL_{self.bb_period}_{self.bb_std}"
        upper_col = f"BBU_{self.bb_period}_{self.bb_std}"
        mid_col = f"BBM_{self.bb_period}_{self.bb_std}"
        for col in (lower_col, upper_col, mid_col):
            if col not in bb.columns:
                return None

        try:
            lower = bb[lower_col].iloc[-1]
            upper = bb[upper_col].iloc[-1]
            mid = bb[mid_col].iloc[-1]
            price = close.iloc[-1]
            rsi_val = rsi.iloc[-1]
        except (IndexError, KeyError, TypeError):
            return None

        if pd.isna(lower) or pd.isna(upper) or pd.isna(mid) or pd.isna(price) or pd.isna(rsi_val):
            return None

        if price <= lower and rsi_val < self.rsi_oversold:
            pct_below = (lower - price) / lower if lower != 0 else 0
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

        if price >= upper and rsi_val > self.rsi_overbought:
            pct_above = (price - upper) / upper if upper != 0 else 0
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
        # Guard against None or empty DataFrame
        if df is None or df.empty or "close" not in df.columns:
            empty_series = pd.Series(False, index=df.index if df is not None else [])
            return BacktestSignals(entries=empty_series, exits=empty_series)

        close = df["close"]
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        rsi = ta.rsi(close, length=self.rsi_period)

        # If indicators could not be computed, return empty signals
        if bb is None or rsi is None:
            empty_series = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        lower_col = f"BBL_{self.bb_period}_{self.bb_std}"
        upper_col = f"BBU_{self.bb_period}_{self.bb_std}"
        mid_col = f"BBM_{self.bb_period}_{self.bb_std}"
        required_cols = {lower_col, upper_col, mid_col}
        if not required_cols.issubset(bb.columns):
            empty_series = pd.Series(False, index=df.index)
            return BacktestSignals(entries=empty_series, exits=empty_series)

        lower = bb[lower_col].shift(1)
        upper = bb[upper_col].shift(1)
        mid = bb[mid_col].shift(1)
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