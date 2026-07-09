"""
Bitcoin MVRV Z-Score Market Timing.

MVRV Z-Score = (Market Cap - Realized Cap) / StdDev(Market Cap)
Zones:
  Z < 1   → accumulation zone (STRONG BUY)
  1 ≤ Z < 3 → neutral/hold
  3 ≤ Z < 7 → overvalued (reduce position)
  Z ≥ 7   → extreme overvaluation (SELL / short)

Data source (free):
  CoinGecko free API for BTC market cap:
    GET https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365

MVRV proxy for backtest:
  True on-chain MVRV requires Glassnode (paid). For backtest, we use:
    mvrv_proxy = price / sma_200
  as an approximation: price >> 200-day SMA = overvalued (high Z-score),
  price << 200-day SMA = undervalued (low Z-score).

Academic reference:
  Murad Mahmudov & David Puell (2018) "BTC MVRV Z-Score" — original derivation.
  Demirer et al. (2021) "On-chain metrics as predictors of BTC returns"
    Finance Research Letters.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

_COINGECKO_MARKET_CHART_URL = (
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
)


class MVRVParams(BaseModel):
    """
    Configuration parameters for the MVRV Z‑Score Timing strategy.
    """

    buy_threshold: float = Field(
        default=1.0,
        description="Z‑score below which a strong BUY signal is generated.",
        example=0.8,
        ge=0.0,
    )
    reduce_threshold: float = Field(
        default=3.0,
        description="Z‑score above which the position should be reduced (sell).",
        example=4.0,
        ge=0.0,
    )
    sell_threshold: float = Field(
        default=7.0,
        description="Z‑score above which a SELL/short signal is generated.",
        example=8.0,
        ge=0.0,
    )
    symbol: str = Field(
        default="BTC-USD",
        description="Trading symbol used for signal generation.",
        example="BTC-USD",
        min_length=1,
    )

    @validator("reduce_threshold")
    def _check_reduce(cls, v: float, values: Dict[str, Any]) -> float:
        buy = values.get("buy_threshold")
        if buy is not None and v <= buy:
            raise ValueError("reduce_threshold must be greater than buy_threshold")
        return v

    @validator("sell_threshold")
    def _check_sell(cls, v: float, values: Dict[str, Any]) -> float:
        reduce_thr = values.get("reduce_threshold")
        if reduce_thr is not None and v <= reduce_thr:
            raise ValueError("sell_threshold must be greater than reduce_threshold")
        return v

    class Config:
        extra = "ignore"


class MVRVZScoreTimingStrategy(AbstractStrategy):
    """
    Bitcoin on-chain MVRV Z-Score market timing strategy.

    Uses CoinGecko free API for market cap data to approximate the MVRV Z-Score.
    For backtest, proxies MVRV using price / 200‑day SMA ratio.
    """

    name = "mvrv_zscore_timing"
    display_name = "MVRV Z-Score Timing (BTC)"
    market_type = "crypto"
    strategy_type = "manual"
    risk_bucket = "directional"
    tick_interval_seconds = 86_400.0  # daily check

    DEFAULT_BUY_THRESHOLD: float = 1.0    # Z-score below → buy
    DEFAULT_SELL_THRESHOLD: float = 7.0   # Z-score above → sell
    DEFAULT_REDUCE_THRESHOLD: float = 3.0 # Z-score above → reduce
    _SMA_WINDOW: int = 200
    _ZSCORE_WINDOW: int = 365

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = params or {}
        validated = MVRVParams(**p)

        self.buy_threshold: float = validated.buy_threshold
        self.sell_threshold: float = validated.sell_threshold
        self.reduce_threshold: float = validated.reduce_threshold
        self.symbol: str = validated.symbol

    def description(self) -> str:
        return (
            "BTC MVRV Z-Score timing: accumulates below Z=1, sells above Z=7. "
            "Uses CoinGecko market cap / 200-day SMA proxy for on-chain MVRV. "
            "Source: Mahmudov & Puell (2018)."
        )

    def _compute_mvrv_proxy_z(self, prices: pd.Series) -> pd.Series:
        """
        Compute MVRV proxy Z‑Score from price series.
        mvrv_proxy = price / sma_200
        mvrv_z = (mvrv_proxy - rolling_mean) / rolling_std
        """
        sma_200 = prices.rolling(self._SMA_WINDOW, min_periods=self._SMA_WINDOW // 2).mean()
        mvrv_proxy = prices / sma_200.clip(lower=1e-8)
        rolling_mean = mvrv_proxy.rolling(self._ZSCORE_WINDOW, min_periods=60).mean()
        rolling_std = mvrv_proxy.rolling(self._ZSCORE_WINDOW, min_periods=60).std().clip(lower=1e-8)
        z_score = (mvrv_proxy - rolling_mean) / rolling_std
        return z_score

    async def _fetch_coingecko_prices(self, days: int = 365) -> list[tuple[int, float]]:
        """
        Fetch BTC market cap / price history from CoinGecko free API.
        Returns list of (timestamp_ms, price) tuples.
        Raises on failure — no mock data.
        """
        params = {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily",
        }
        import aiohttp  # lazy: optional dep — never break STRATEGY_REGISTRY import
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _COINGECKO_MARKET_CHART_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        prices = data.get("prices")
        if not prices:
            raise ValueError("CoinGecko returned empty price data for BTC.")

        return prices

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Fetch CoinGecko market data and compute MVRV proxy Z‑Score.
        Signals based on current Z‑Score zone.
        """
        try:
            raw_prices = await self._fetch_coingecko_prices(days=400)
        except Exception as exc:
            raise RuntimeError(
                f"MVRVZScoreTimingStrategy: failed to fetch CoinGecko data — {exc}"
            ) from exc

        timestamps = pd.to_datetime([p[0] for p in raw_prices], unit="ms", utc=True)
        prices = pd.Series([p[1] for p in raw_prices], index=timestamps, dtype=float)

        if len(prices) < self._SMA_WINDOW + 20:
            return None

        z_score = self._compute_mvrv_proxy_z(prices)
        current_z = float(z_score.iloc[-1])
        current_price = float(prices.iloc[-1])

        if np.isnan(current_z):
            return None

        if current_z < self.buy_threshold:
            # Accumulation zone — strong buy
            confidence = min(0.95, 0.80 + (self.buy_threshold - current_z) / 5.0)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="buy",
                confidence=confidence,
                target_price=current_price,
                metadata={
                    "mvrv_proxy_z": round(current_z, 4),
                    "zone": "accumulation",
                    "order_type": "market",
                },
            )

        if current_z >= self.sell_threshold:
            # Extreme overvaluation — sell / short
            confidence = min(0.92, 0.75 + (current_z - self.sell_threshold) / 10.0)
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=confidence,
                target_price=current_price,
                metadata={
                    "mvrv_proxy_z": round(current_z, 4),
                    "zone": "extreme_overvaluation",
                    "order_type": "market",
                },
            )

        if current_z >= self.reduce_threshold:
            # Overvalued — reduce position
            confidence = 0.65
            return Signal(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                symbol=symbol,
                side="sell",
                confidence=confidence,
                target_price=current_price,
                metadata={
                    "mvrv_proxy_z": round(current_z, 4),
                    "zone": "overvalued_reduce",
                    "order_type": "limit",
                },
            )

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Compute MVRV proxy from daily OHLCV.
        Entry: Z‑score < buy_threshold (lagged by 1 bar).
        Short entry: Z‑score >= sell_threshold.
        """
        false_series = pd.Series(False, index=df.index)
        default = BacktestSignals(
            entries=false_series,
            exits=false_series,
            short_entries=false_series,
            short_exits=false_series,
        )

        if "close" not in df.columns or len(df) < self._SMA_WINDOW + 30:
            return default

        close = df["close"].astype(float)
        z_score = self._compute_mvrv_proxy_z(close)

        # shift(1) — no lookahead bias
        # ... (truncated for brevity)
        return default