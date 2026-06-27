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
from typing import List

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

    async def compute_features_async(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add funding rate and open interest features to an OHLCV DataFrame.

        df must have a DatetimeIndex (UTC). Adds columns:
          funding_rate        — most recent funding rate (8h)
          funding_rate_ma7    — 7‑day MA of funding rate
          oi_change_pct       — day‑over‑day OI % change
          oi_momentum         — 7‑day OI momentum (current / MA7 - 1)

        Missing data → NaN (not filled with fake values).
        """
        df = df.copy()
        # initialise columns with NaN
        for col in ("funding_rate", "funding_rate_ma7", "oi_change_pct", "oi_momentum"):
            df[col] = np.nan

        fr_df, oi_df = await asyncio.gather(
            self.get_funding_rate_history(symbol, limit=500),
            self.get_open_interest_hist(symbol, period="1d", limit=500),
        )

        # ---- Funding Rate ----------------------------------------------------
        if not fr_df.empty:
            fr_df = (
                fr_df.set_index("ts")
                .sort_index()
                .resample("D")
                .last()
                .rename(columns={"funding_rate": "funding_rate"})
            )
            fr_df["funding_rate_ma7"] = fr_df["funding_rate"].rolling(7, min_periods=1).mean()

            # Align to the OHLCV index (date‑only)
            idx = df.index.normalize()
            df["funding_rate"] = idx.map(fr_df["funding_rate"])
            df["funding_rate_ma7"] = idx.map(fr_df["funding_rate_ma7"])

        # ---- Open Interest ----------------------------------------------------
        if not oi_df.empty:
            oi_df = (
                oi_df.set_index("ts")
                .sort_index()
                .resample("D")
                .last()
                .rename(columns={"open_interest": "open_interest"})
            )
            oi_df["oi_change_pct"] = oi_df["open_interest"].pct_change() * 100
            oi_df["oi_momentum"] = (
                oi_df["open_interest"] / oi_df["open_interest"].rolling(7, min_periods=1).mean()
                - 1
            )

            idx = df.index.normalize()
            df["oi_change_pct"] = idx.map(oi_df["oi_change_pct"])
            df["oi_momentum"] = idx.map(oi_df["oi_momentum"])

        return df

    def compute_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Sync wrapper — runs the async version via asyncio."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Already inside an event loop (e.g., FastAPI) — offload to a thread
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


def generate_alternative_signal(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals based on alternative features.

    Returns a pandas Series indexed like ``df`` with values:
        1  – long entry / hold
       -1  – short entry / hold
        0  – flat / exit

    Entry logic (tightened):
        • Long when:
            - funding_rate > funding_rate_ma7 (positive funding pressure)
            - oi_momentum > 0 (OI above its 7‑day MA)
            - oi_change_pct > 0 (day‑over‑day OI increase)
            - price trend confirmed by a 20‑period SMA (close > SMA20)
        • Short when the opposite holds.

    Confirmation filters:
        • Both funding and OI conditions must be satisfied for at least two consecutive days.
        • A minimum absolute funding spread of 0.0001 is required to avoid noise.

    Exit logic:
        • Exit long when funding_rate drops below its MA or oi_momentum turns negative.
        • Exit short when funding_rate rises above its MA or oi_momentum turns positive.
        • Additionally, exit when the price crosses the opposite SMA20 direction.

    The function gracefully handles missing data – NaN values yield a flat signal (0).
    """
    required_cols = {"funding_rate", "funding_rate_ma7", "oi_change_pct", "oi_momentum", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns for signal generation: {missing}")

    df = df.copy()

    # Price trend SMA20
    df["sma20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # Basic directional flags
    df["funding_up"] = (df["funding_rate"] - df["funding_rate_ma7"]).gt(0.0001)
    df["funding_down"] = (df["funding_rate_ma7"] - df["funding_rate"]).gt(0.0001)

    df["oi_pos"] = (df["oi_momentum"] > 0) & (df["oi_change_pct"] > 0)
    df["oi_neg"] = (df["oi_momentum"] < 0) & (df["oi_change_pct"] < 0)

    df["price_up"] = df["close"] > df["sma20"]
    df["price_down"] = df["close"] < df["sma20"]

    # Confirmation: require two consecutive true days
    df["funding_up_2"] = df["funding_up"] & df["funding_up"].shift(1).fillna(False)
    df["funding_down_2"] = df["funding_down"] & df["funding_down"].shift(1).fillna(False)

    df["oi_pos_2"] = df["oi_pos"] & df["oi_pos"].shift(1).fillna(False)
    df["oi_neg_2"] = df["oi_neg"] & df["oi_neg"].shift(1).fillna(False)

    # Long entry condition
    long_entry = df["funding_up_2"] & df["oi_pos_2"] & df["price_up"]
    # Short entry condition
    short_entry = df["funding_down_2"] & df["oi_neg_2"] & df["price_down"]

    # Build raw signal
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[long_entry] = 1
    signal[short_entry] = -1

    # Propagate position forward until an exit condition occurs
    position = 0
    positions = []
    for idx, row in df.iterrows():
        if position == 0:
            # no open position – adopt fresh signal if any
            position = signal.at[idx]
        else:
            # check exit criteria for the current side
            if position == 1:
                exit_cond = (
                    (row["funding_rate"] <= row["funding_rate_ma7"])
                    | (row["oi_momentum"] <= 0)
                    | (row["close"] <= row["sma20"])
                )
                if exit_cond:
                    position = 0
            elif position == -1:
                exit_cond = (
                    (row["funding_rate"] >= row["funding_rate_ma7"])
                    | (row["oi_momentum"] >= 0)
                    | (row["close"] >= row["sma20"])
                )
                if exit_cond:
                    position = 0
        positions.append(position)

    return pd.Series(positions, index=df.index, dtype=int)