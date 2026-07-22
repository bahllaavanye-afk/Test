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

import httpx
import numpy as np
import pandas as pd

_FAPI_BASE = "https://fapi.binance.com"
_FUTURES_DATA_BASE = "https://fapi.binance.com"


def _to_binance_symbol(symbol: str) -> str:
    """Convert 'BTC-USD' or 'BTC/USDT' to 'BTCUSDT'."""
    return symbol.replace("-", "").replace("/", "").upper()


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
        bn_sym = _to_binance_symbol(symbol)
        url = f"{_FAPI_BASE}/fapi/v1/fundingRate"
        params = {"symbol": bn_sym, "limit": min(limit, 1000)}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
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
        bn_sym = _to_binance_symbol(symbol)
        url = f"{_FUTURES_DATA_BASE}/futures/data/openInterestHist"
        params = {"symbol": bn_sym, "period": period, "limit": min(limit, 500)}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
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
            oi_df["oi_momentum"] = oi_df["open_interest"] / oi_df["open_interest"].rolling(7).mean() - 1

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
                    future = pool.submit(asyncio.run, self.compute_features_async(symbol, df))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.compute_features_async(symbol, df))
        except Exception:
            df = df.copy()
            for col in ("funding_rate", "funding_rate_ma7", "oi_change_pct", "oi_momentum"):
                df[col] = np.nan
            return df


ALTERNATIVE_FEATURE_COLS = [
    "funding_rate",
    "funding_rate_ma7",
    "oi_change_pct",
    "oi_momentum",
]

_binance_features = BinanceFundingRateFeatures()


def add_alternative_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Add Binance alternative data features for crypto symbols.
    For non-crypto symbols, adds columns filled with NaN.
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


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals using alternative data features.

    Signal logic (tightened entry conditions):
      • Long entry (signal = 1):
          - funding_rate < funding_rate_ma7 (funding cheaper than recent average)
          - oi_momentum > 0 (open interest rising)
          - oi_change_pct > 0 (day‑over‑day OI increase)
          - All three conditions must hold simultaneously.

      • Short entry (signal = -1):
          - funding_rate > funding_rate_ma7
          - oi_momentum < 0
          - oi_change_pct < 0
          - All three conditions must hold simultaneously.

    Exit logic (signal = 0):
      • Existing position is closed when any of the entry conditions reverses.
      • If no position is open, signal remains 0.

    The function adds a ``signal`` column to the DataFrame and returns it.
    Missing feature values are treated as no‑signal (0).
    """
    df = df.copy()

    # Ensure required columns exist
    required = {"funding_rate", "funding_rate_ma7", "oi_momentum", "oi_change_pct"}
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Boolean masks for entry conditions
    long_entry = (
        (df["funding_rate"] < df["funding_rate_ma7"])
        & (df["oi_momentum"] > 0)
        & (df["oi_change_pct"] > 0)
    )
    short_entry = (
        (df["funding_rate"] > df["funding_rate_ma7"])
        & (df["oi_momentum"] < 0)
        & (df["oi_change_pct"] < 0)
    )

    # Initialise signal column with zeros
    df["signal"] = 0

    # Apply entry signals where data is not missing
    valid_long = long_entry & df[["funding_rate", "funding_rate_ma7", "oi_momentum", "oi_change_pct"]].notna().all(axis=1)
    valid_short = short_entry & df[["funding_rate", "funding_rate_ma7", "oi_momentum", "oi_change_pct"]].notna().all(axis=1)

    df.loc[valid_long, "signal"] = 1
    df.loc[valid_short, "signal"] = -1

    # Implement exit logic: when a position exists, close if any condition flips
    # Forward‑fill the position to know the current exposure
    df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)

    exit_condition = (
        (df["position"] == 1) & ~valid_long
    ) | (
        (df["position"] == -1) & ~valid_short
    )

    df.loc[exit_condition, "signal"] = 0
    df["position"] = df["signal"].replace(to_replace=0, method="ffill").fillna(0)

    # Clean up intermediate columns
    df.drop(columns=["position"], inplace=True)

    return df