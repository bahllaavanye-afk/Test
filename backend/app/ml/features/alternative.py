"""
Alternative data features from free public sources.

BinanceFundingRateFeatures:
  - Funding rates from Binance Futures (public, no auth)
  - Open interest history (public, no auth)
  - Features: funding_rate, funding_rate_ma7, oi_change_pct, oi_momentum

All API calls are async. For sync contexts, use compute_features_sync().
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal, List

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

_FAPI_BASE = "https://fapi.binance.com"
_FUTURES_DATA_BASE = "https://fapi.binance.com"


def _to_binance_symbol(symbol: str) -> str:
    """Convert 'BTC-USD' or 'BTC/USDT' to 'BTCUSDT'."""
    return symbol.replace("-", "").replace("/", "").upper()


class FundingRateParams(BaseModel):
    """Parameters for requesting Binance funding rate history."""

    symbol: str = Field(
        ...,
        description="Trading pair symbol, e.g., 'BTC-USD' or 'BTC/USDT'.",
        example="BTC-USD",
    )
    limit: int = Field(
        500,
        ge=1,
        le=1000,
        description="Maximum number of funding rate records to fetch (max 1000).",
        example=500,
    )

    @validator("symbol")
    def symbol_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must be a non‑empty string")
        return v


class OpenInterestParams(BaseModel):
    """Parameters for requesting Binance open interest history."""

    symbol: str = Field(
        ...,
        description="Trading pair symbol, e.g., 'BTC-USD' or 'BTC/USDT'.",
        example="BTC-USD",
    )
    period: Literal[
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
    ] = Field(
        "1d",
        description="Kline interval period for open‑interest data.",
        example="1d",
    )
    limit: int = Field(
        500,
        ge=1,
        le=500,
        description="Maximum number of open‑interest records to fetch (max 500).",
        example=500,
    )

    @validator("symbol")
    def symbol_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must be a non‑empty string")
        return v


class ComputeFeaturesParams(BaseModel):
    """Parameters for computing alternative features on an OHLCV DataFrame."""

    symbol: str = Field(
        ...,
        description="Trading pair symbol, e.g., 'BTC-USD' or 'BTC/USDT'.",
        example="BTC-USD",
    )
    df: pd.DataFrame = Field(
        ...,
        description="OHLCV DataFrame with a UTC DatetimeIndex.",
        example="pd.DataFrame(...)",
    )

    @validator("symbol")
    def symbol_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must be a non‑empty string")
        return v

    @validator("df")
    def df_must_have_datetime_index(cls, v: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if v.index.tz is None:
            raise ValueError("df DatetimeIndex must be timezone‑aware (UTC)")
        return v


class BinanceFundingRateFeatures:
    """
    Pull Binance Futures funding rates + open interest.
    Binance public endpoints — no API key required.
    """

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        GET /fapi/v1/fundingRate

        Returns DataFrame with columns: [ts, funding_rate] sorted ascending.
        Returns empty DataFrame on any error.
        """
        # Validate inputs via Pydantic
        params = FundingRateParams(symbol=symbol, limit=limit)

        bn_sym = _to_binance_symbol(params.symbol)
        url = f"{_FAPI_BASE}/fapi/v1/fundingRate"
        request_params = {"symbol": bn_sym, "limit": min(params.limit, 1000)}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=request_params)
                resp.raise_for_status()
                data = resp.json()
            if not data:
                return pd.DataFrame()
            rows = [
                {
                    "ts": pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True),
                    "funding_rate": float(r["fundingRate"]),
                }
                for r in data
            ]
            df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        GET /futures/data/openInterestHist

        Returns DataFrame with columns: [ts, open_interest, open_interest_value].
        Returns empty DataFrame on any error.
        Valid periods: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        """
        # Validate inputs via Pydantic
        params = OpenInterestParams(symbol=symbol, period=period, limit=limit)

        bn_sym = _to_binance_symbol(params.symbol)
        url = f"{_FUTURES_DATA_BASE}/futures/data/openInterestHist"
        request_params = {
            "symbol": bn_sym,
            "period": params.period,
            "limit": min(params.limit, 500),
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=request_params)
                resp.raise_for_status()
                data = resp.json()
            if not data:
                return pd.DataFrame()
            rows = [
                {
                    "ts": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                    "open_interest": float(r["sumOpenInterest"]),
                    "open_interest_value": float(r["sumOpenInterestValue"]),
                }
                for r in data
            ]
            df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    async def compute_features_async(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add funding rate and open interest features to an OHLCV DataFrame.

        df must have a DatetimeIndex (UTC). Adds columns:
          funding_rate        — most recent funding rate (8h)
          funding_rate_ma7    — 7-period MA of funding rate
          oi_change_pct       — day-over-day OI % change
          oi_momentum         — 7-day OI momentum (current / MA7 - 1)

        Missing data → NaN (not filled with fake values).
        """
        # Validate via Pydantic model
        ComputeFeaturesParams(symbol=symbol, df=df)

        df = df.copy()
        for col in ("funding_rate", "funding_rate_ma7", "oi_change_pct", "oi_momentum"):
            df[col] = np.nan

        fr_df, oi_df = await asyncio.gather(
            self.get_funding_rate_history(symbol, limit=500),
            self.get_open_interest_hist(symbol, period="1d", limit=500),
        )

        # Merge funding rate
        if not fr_df.empty:
            fr_df = fr_df.set_index("ts")
            fr_df = fr_df.resample("D").last()  # one value per day
            fr_df["funding_rate_ma7"] = fr_df["funding_rate"].rolling(7).mean()

            if hasattr(df.index, "tz") and df.index.tz is not None:
                idx = df.index.normalize()
            else:
                idx = pd.to_datetime(df.index).tz_localize("UTC").normalize()

            for i, ts in enumerate(idx):
                ts_day = ts.normalize()
                if ts_day in fr_df.index:
                    df.iloc[i, df.columns.get_loc("funding_rate")] = float(
                        fr_df.loc[ts_day, "funding_rate"]
                    )
                    df.iloc[i, df.columns.get_loc("funding_rate_ma7")] = float(
                        fr_df.loc[ts_day, "funding_rate_ma7"]
                    )

        # Merge OI
        if not oi_df.empty:
            oi_df = oi_df.set_index("ts").resample("D").last()
            oi_df["oi_change_pct"] = oi_df["open_interest"].pct_change() * 100
            oi_df["oi_momentum"] = (
                oi_df["open_interest"] / oi_df["open_interest"].rolling(7).mean() - 1
            )

            if hasattr(df.index, "tz") and df.index.tz is not None:
                idx = df.index.normalize()
            else:
                idx = pd.to_datetime(df.index).tz_localize("UTC").normalize()

            for i, ts in enumerate(idx):
                ts_day = ts.normalize()
                if ts_day in oi_df.index:
                    df.iloc[i, df.columns.get_loc("oi_change_pct")] = float(
                        oi_df.loc[ts_day, "oi_change_pct"]
                    )
                    df.iloc[i, df.columns.get_loc("oi_momentum")] = float(
                        oi_df.loc[ts_day, "oi_momentum"]
                    )

        return df

    def compute_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Sync wrapper — runs the async version via asyncio."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Already inside an event loop (e.g., FastAPI) — create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.compute_features_async(symbol, df)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.compute_features_async(symbol, df))
        except Exception:
            df = df.copy()
            for col in ("funding_rate", "funding_rate_ma7", "oi_change_pct", "oi_momentum"):
                df[col] = np.nan
            return df


ALTERNATIVE_FEATURE_COLS: List[str] = [
    "funding_rate",
    "funding_rate_ma7",
    "oi_change_pct",
    "oi_momentum",
]

_binance_features = BinanceFundingRateFeatures()


def add_alternative_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Add Binance alternative data features for crypto symbols.
    For non‑crypto symbols, adds columns filled with NaN.
    """
    is_crypto = any(
        kw in symbol.upper()
        for kw in ("BTC", "ETH", "BNB", "SOL", "XRP", "USDT", "USDC", "CRYPTO")
    )
    if is_crypto and symbol:
        return _binance_features.compute_features(symbol, df)

    df = df.copy()
    for col in ALTERNATIVE_FEATURE_COLS:
        df[col] = np.nan
    return df