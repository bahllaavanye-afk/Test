"""
Crypto Intraday Seasonality Strategy.

Systematic studies of Bitcoin 1-hour returns show persistent calendar patterns:
  - 22:00 UTC hour: average +0.07% return — highest of the 24 hours.
  - European session (07:00-15:00 UTC): above-average hourly returns.
  - Asian session overnight (01:00-06:00 UTC): below-average / negative.

Strategy:
  Primary window  : buy at 21:45 UTC, sell at 22:15 UTC (30-min capture)
  Secondary window: buy at 07:00 UTC, sell at 15:00 UTC (European session)

Backtest (1-hour OHLCV from yfinance): return +1 during peak hours, 0 otherwise.

Academic reference:
  Baur, Cahill, Godfrey & Liu (2019) "Bitcoin Time-of-Day, Day-of-Week, and
  Month-of-Year Effects in Returns and Trading Volume" — Finance Research
  Letters, 31, 78-92.
  Petukhina et al. (2021) "Investing with Cryptocurrencies — evaluating their
  potential for portfolio allocation strategies" — Quantitative Finance.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class IntradaySeasonality(AbstractStrategy):
    """
    Crypto intraday seasonality: trade peak-return UTC hours.

    peak_hours    : list of UTC hours with highest average returns.
    secondary_hours: range of UTC hours for the European session trade.
    """

    name = "intraday_seasonality"
    display_name = "Crypto Intraday Seasonality"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0  # 15-minute polling

    # Default seasonality windows (UTC hours)
    DEFAULT_PEAK_HOURS: list[int] = [22]
    DEFAULT_SECONDARY_START: int = 7
    DEFAULT_SECONDARY_END: int = 15  # exclusive

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        self.peak_hours: list[int] = list(p.get("peak_hours", self.DEFAULT_PEAK_HOURS))
        self.secondary_start: int = int(p.get("secondary_start", self.DEFAULT_SECONDARY_START))
        self.secondary_end: int = int(p.get("secondary_end", self.DEFAULT_SECONDARY_END))
        self.secondary_hours: range = range(self.secondary_start, self.secondary_end)

    def description(self) -> str:
        return (
            "Trades BTC intraday seasonality: long during 22:00 UTC peak-return hour "
            "and European session 07:00-15:00 UTC. "
            "Source: Baur et al. (2019) 'Bitcoin Time-of-Day Effects'."
        )

    def _is_peak_hour(self, utc_hour: int) -> bool:
        return utc_hour in self.peak_hours

    def _is_secondary_hour(self, utc_hour: int) -> bool:
        return utc_hour in self.secondary_hours

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Check current UTC hour and emit BUY when entering a peak window.
        """
        now_utc = datetime.now(UTC)
        utc_hour = now_utc.hour
        utc_minute = now_utc.minute

        if "close" not in data.columns or len(data) == 0:
            raise ValueError("IntradaySeasonality.analyze: 'close' column required in data.")

        current_price = float(data["close"].iloc[-1])
        if current_price <= 0:
            raise ValueError(f"IntradaySeasonality: invalid price {current_price}")

        # Primary: enter at 21:45 to capture the 22:00 peak
        if utc_hour == 21 and utc_minute >= 45:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=0.72,
                target_price=current_price,
                metadata={
                    "window": "primary_peak_entry",
                    "utc_hour": utc_hour,
                    "utc_minute": utc_minute,
                    "order_type": "market",
                },
            )

        # Primary: exit at 22:15
        if utc_hour == 22 and utc_minute >= 15:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=0.72,
                target_price=current_price,
                metadata={
                    "window": "primary_peak_exit",
                    "utc_hour": utc_hour,
                    "order_type": "market",
                },
            )

        # Secondary: enter European session at 07:00
        if utc_hour == 7 and utc_minute < 15:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=0.65,
                target_price=current_price,
                metadata={
                    "window": "european_session_entry",
                    "order_type": "market",
                },
            )

        # Secondary: exit European session at 15:00
        if utc_hour == 15 and utc_minute < 15:
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=0.65,
                target_price=current_price,
                metadata={
                    "window": "european_session_exit",
                    "order_type": "market",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Use 1-hour OHLCV bars. Signal = +1 during peak/secondary hours, 0 otherwise.
        Index must be a DatetimeIndex with UTC timezone (or timezone-naive UTC).
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < 24:
            return default

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return default

        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        utc_hours = pd.Series(idx.hour, index=df.index)

        # In-window: peak hours or secondary hours
        in_peak = utc_hours.isin(self.peak_hours)
        in_secondary = utc_hours.isin(list(self.secondary_hours))
        in_window = in_peak | in_secondary

        # Entry: first bar entering the window
        # shift(1) to prevent lookahead — yesterday's signal drives today's trade
        in_window_lag = in_window.shift(1).fillna(False).astype(bool)
        not_in_window_lag = (~in_window).shift(1).fillna(True).astype(bool)

        entries = in_window_lag
        exits = not_in_window_lag

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=false_series,
            short_exits=false_series,
        )
