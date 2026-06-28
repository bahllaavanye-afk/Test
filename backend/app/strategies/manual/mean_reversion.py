"""
Bollinger Band Mean Reversion Strategy.
Enter when price touches lower/upper band; exit at middle band.
"""
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


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

    @staticmethod
    def _bollinger_bands(series: pd.Series, period: int, std_mult: float):
        """
        Compute Bollinger Bands (lower, middle, upper) using rolling statistics.
        Returns a DataFrame with columns ['lower', 'middle', 'upper'].
        """
        rolling_mean = series.rolling(window=period, min_periods=period).mean()
        rolling_std = series.rolling(window=period, min_periods=period).std()
        middle = rolling_mean
        upper = rolling_mean + std_mult * rolling_std
        lower = rolling_mean - std_mult * rolling_std
        return pd.DataFrame({"lower": lower, "middle": middle, "upper": upper}, index=series.index)

    @staticmethod
    def _rsi(series: pd.Series, period: int):
        """
        Compute Relative Strength Index using the standard EMA smoothing method.
        Returns a Series aligned with the input series.
        """
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        # Use exponential weighted moving average for smoothing
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyze the latest data point to generate a trading signal.
        Optimized to compute only the required recent window for Bollinger Bands and RSI.
        """
        if "close" not in data.columns or len(data) < self.bb_period + 5:
            return None

        close = data["close"]

        # Slice only the recent window needed for the latest Bollinger values
        recent_close = close.iloc[-self.bb_period - 1 :]  # include one extra for shift safety

        bb_df = self._bollinger_bands(recent_close, self.bb_period, self.bb_std)
        rsi_series = self._rsi(recent_close, self.rsi_period)

        # Ensure we have at least one non‑NaN value after the calculations
        if bb_df.isnull().all().any() or rsi_series.isnull().all():
            return None

        lower = bb_df["lower"].iloc[-1]
        upper = bb_df["upper"].iloc[-1]
        mid = bb_df["middle"].iloc[-1]
        price = close.iloc[-1]
        rsi_val = rsi_series.iloc[-1]

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
        Generate backtest signals for the entire dataset.
        Utilizes vectorized pandas operations for efficiency.
        """
        close = df["close"]
        bb_df = self._bollinger_bands(close, self.bb_period, self.bb_std)
        rsi_series = self._rsi(close, self.rsi_period)

        # Shift by one period to use previous bar's indicators for entry decisions
        lower = bb_df["lower"].shift(1)
        upper = bb_df["upper"].shift(1)
        mid = bb_df["middle"].shift(1)
        rsi_s = rsi_series.shift(1)
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