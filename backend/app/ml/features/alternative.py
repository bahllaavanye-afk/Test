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
from typing import List

import httpx
import numpy as np
import pandas as pd

_FAPI_BASE = "https://fapi.binance.com"
_FUTURES_DATA_BASE = "https://fapi.binance.com"


def _to_binance_symbol(symbol: str) -> str:
    """Convert a generic symbol (e.g., ``BTC-USD`` or ``BTC/USDT``) to Binance's format.

    Args:
        symbol: The input trading symbol.

    Returns:
        A string with hyphens and slashes removed and upper‑cased, e.g. ``BTCUSDT``.
    """
    return symbol.replace("-", "").replace("/", "").upper()


class BinanceFundingRateFeatures:
    """
    Pull Binance Futures funding rates and open interest.

    Binance public endpoints are used; no API key or authentication is required.
    """

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Retrieve historical funding rates for a given symbol.

        Calls the ``GET /fapi/v1/fundingRate`` endpoint.

        Args:
            symbol: Trading pair symbol (e.g., ``BTCUSDT``).
            limit: Maximum number of rows to fetch (capped at 1000 by the API).

        Returns:
            A ``pandas.DataFrame`` with columns ``[ts, funding_rate]`` sorted
            ascending by timestamp. Returns an empty ``DataFrame`` on any error.
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
        Retrieve open‑interest history for a given symbol.

        Calls the ``GET /futures/data/openInterestHist`` endpoint.

        Args:
            symbol: Trading pair symbol (e.g., ``BTCUSDT``).
            period: Aggregation period accepted by Binance (e.g., ``1d``).
            limit: Maximum number of rows to fetch (capped at 500 by the API).

        Returns:
            A ``pandas.DataFrame`` with columns ``[ts, open_interest, open_interest_value]``
            sorted ascending by timestamp. Returns an empty ``DataFrame`` on any error.
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
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add funding‑rate and open‑interest features to an OHLCV ``DataFrame``.

        The input ``df`` must have a ``DatetimeIndex`` (UTC). Four new columns are added:

        - ``funding_rate`` – most recent 8‑hour funding rate.
        - ``funding_rate_ma7`` – 7‑day moving average of the funding rate.
        - ``oi_change_pct`` – day‑over‑day open‑interest percentage change.
        - ``oi_momentum`` – 7‑day open‑interest momentum (current / MA7 - 1).

        Missing data are left as ``NaN``; no imputation is performed.

        Args:
            symbol: Trading pair symbol (e.g., ``BTCUSDT``).
            df: OHLCV ``DataFrame`` with a UTC ``DatetimeIndex``.

        Returns:
            A copy of ``df`` with the four alternative‑data columns appended.
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
        """
        Synchronous wrapper that executes the asynchronous feature computation.

        This method can be called from regular (non‑async) code. It attempts to
        reuse an existing event loop when possible; otherwise it creates a new loop.
        In case of any failure, the original ``df`` is returned with the feature
        columns added but filled with ``NaN`` values.

        Args:
            symbol: Trading pair symbol (e.g., ``BTCUSDT``).
            df: OHLCV ``DataFrame`` with a ``DatetimeIndex``.

        Returns:
            A ``DataFrame`` containing the original data plus the alternative‑data
            columns (or ``NaN`` values if the computation failed).
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Already inside an event loop (e.g., FastAPI) — run in a thread.
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
    Append Binance alternative‑data features to a DataFrame.

    If ``symbol`` appears to be a cryptocurrency (based on a simple keyword
    heuristic), the function fetches funding‑rate and open‑interest data from
    Binance and merges the resulting features. For non‑crypto symbols, the
    expected feature columns are added but populated with ``NaN``.

    Args:
        df: Input ``DataFrame`` (typically OHLCV) to which the features will be added.
        symbol: Trading symbol to identify whether to query Binance data.

    Returns:
        A new ``DataFrame`` containing the original columns plus the
        alternative‑data columns.
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