from datetime import datetime, timezone
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base  # noqa: F401 — re-export for all models
import pandas as pd
from typing import Sequence, Optional


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class StrategySignalMixin:
    """
    Mixin providing basic signal generation logic.
    Designed to be lightweight and reusable across strategy models.
    """

    # Default parameters – can be overridden in subclasses
    short_window: int = 20
    long_window: int = 50
    volume_threshold: float = 1_000_000.0
    stop_loss_pct: float = 0.02  # 2% stop‑loss

    @classmethod
    def _prepare_dataframe(
        cls,
        prices: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[datetime],
    ) -> pd.DataFrame:
        """
        Convert raw price/volume data into a DataFrame indexed by timestamp.
        Ensures correct dtypes and monotonic index.
        """
        df = pd.DataFrame(
            {"price": prices, "volume": volumes},
            index=pd.to_datetime(timestamps, utc=True),
        )
        df = df.sort_index()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna()
        return df

    @classmethod
    def _moving_average(cls, series: pd.Series, window: int) -> pd.Series:
        """Calculate a simple moving average with a minimum period check."""
        if len(series) < window:
            return pd.Series(dtype=float)
        return series.rolling(window=window, min_periods=window).mean()

    @classmethod
    def is_entry_signal(
        cls,
        prices: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[datetime],
        *,
        short_window: Optional[int] = None,
        long_window: Optional[int] = None,
        volume_threshold: Optional[float] = None,
    ) -> bool:
        """
        Determine if a new long entry signal is present.

        Entry criteria (tightened):
        1. Short‑term MA > Long‑term MA (trend confirmation).
        2. Current volume exceeds the defined threshold.
        3. Price is above the previous close (momentum confirmation).
        """
        short_w = short_window or cls.short_window
        long_w = long_window or cls.long_window
        vol_thr = volume_threshold or cls.volume_threshold

        df = cls._prepare_dataframe(prices, volumes, timestamps)
        if df.empty or len(df) < long_w:
            return False

        df["short_ma"] = cls._moving_average(df["price"], short_w)
        df["long_ma"] = cls._moving_average(df["price"], long_w)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Condition 1: MA crossover
        if not (latest["short_ma"] > latest["long_ma"]):
            return False

        # Condition 2: Volume filter
        if latest["volume"] < vol_thr:
            return False

        # Condition 3: Momentum confirmation
        if latest["price"] <= prev["price"]:
            return False

        return True

    @classmethod
    def is_exit_signal(
        cls,
        entry_price: float,
        prices: Sequence[float],
        timestamps: Sequence[datetime],
        *,
        short_window: Optional[int] = None,
        long_window: Optional[int] = None,
        stop_loss_pct: Optional[float] = None,
    ) -> bool:
        """
        Determine if an exit (close) signal should be triggered.

        Exit criteria (enhanced):
        1. Short‑term MA falls below Long‑term MA (trend reversal).
        2. Price drops below entry_price * (1 - stop_loss_pct) (protective stop‑loss).
        """
        short_w = short_window or cls.short_window
        long_w = long_window or cls.long_window
        sl_pct = stop_loss_pct or cls.stop_loss_pct

        # Ensure we have price data
        if not prices:
            return False

        df = pd.DataFrame(
            {"price": pd.to_numeric(prices, errors="coerce")},
            index=pd.to_datetime(timestamps, utc=True),
        ).sort_index().dropna()

        if len(df) < long_w:
            # Not enough data to compute reliable MAs; rely on stop‑loss only
            latest_price = df["price"].iloc[-1]
            return latest_price <= entry_price * (1 - sl_pct)

        df["short_ma"] = cls._moving_average(df["price"], short_w)
        df["long_ma"] = cls._moving_average(df["price"], long_w)

        latest = df.iloc[-1]

        # Condition 1: MA crossover reversal
        if latest["short_ma"] < latest["long_ma"]:
            return True

        # Condition 2: Stop‑loss breach
        if latest["price"] <= entry_price * (1 - sl_pct):
            return True

        return False